import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNetResNet50
from .base import BaseSegmentationModel
from scipy.optimize import linear_sum_assignment


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim=256, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
    
    def forward(self, x):
        B, C, H, W = x.shape
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).view(H, 1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).view(1, W).repeat(H, 1)
        
        dim_t = torch.arange(self.dim // 4, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t / (self.dim // 4))
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)
        pos_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
        
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos


class TransformerDecoderWithIntermediates(nn.TransformerDecoder):
    def forward(self, tgt, memory, return_intermediate=False):
        output = tgt
        intermediates = []
        
        for mod in self.layers:
            output = mod(output, memory)
            if return_intermediate:
                intermediates.append(output)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return torch.stack(intermediates) if return_intermediate else output


def dice_loss(inputs, targets):
    inputs = torch.sigmoid(inputs).flatten(1)
    targets = targets.flatten(1)
    intersection = (inputs * targets).sum(1)
    union = inputs.sum(1) + targets.sum(1)
    return 1 - (2 * intersection / (union + 1e-8)).mean()


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


class Mask2Former(BaseSegmentationModel):
    def __init__(self, num_classes, emb_dim=256, num_queries=100, nhead=8, nlayers=3, 
                 w_class=2.0, w_dice=5.0, w_focal=5.0, pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_queries = num_queries
        self.w_class = w_class
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.cost_class = 1.0
        self.cost_dice = 1.0
        
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        self.bridge = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec1 = self._make_decoder_block(2048, 1024)
        
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec2 = self._make_decoder_block(1024, 512)
        
        self.feat_proj = nn.Conv2d(512, emb_dim, 1)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._make_decoder_block(512, 256)
        
        self.up4 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec4 = self._make_decoder_block(128, 64)
        
        self.up5 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.pos_emb = PositionEmbedding2D(emb_dim)
        self.queries = nn.Embedding(num_queries, emb_dim)
        self.query_pos = nn.Embedding(num_queries, emb_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, 
            nhead=nhead, 
            dim_feedforward=emb_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerDecoderWithIntermediates(decoder_layer, num_layers=nlayers)
        
        self.class_head = nn.Linear(emb_dim, num_classes + 1)
        self.mask_embed = nn.Linear(emb_dim, emb_dim)
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x):
        B = x.shape[0]
        input_size = x.shape[2:]
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        bridge = self.bridge(enc5)
        
        dec1 = self.up1(bridge)
        dec1 = self._match_size(dec1, enc4)
        dec1 = torch.cat([dec1, enc4], dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = self._match_size(dec2, enc3)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.dec2(dec2)
        
        features = self.feat_proj(dec2)
        pos_emb = self.pos_emb(features)
        
        dec3 = self.up3(dec2)
        dec3 = self._match_size(dec3, enc2)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = self._match_size(dec4, enc1)
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec4 = self.dec4(dec4)
        
        dec5 = self.up5(dec4)
        dec5 = self.dec5(dec5)
        semantic_out = self.final(dec5)
        semantic_out = F.interpolate(semantic_out, size=input_size, mode='bilinear', align_corners=False)
        
        mem = features.flatten(2).permute(0, 2, 1)
        mem_pos = pos_emb.flatten(2).permute(0, 2, 1)
        
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        q_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        
        intermediate_queries = self.transformer(queries + q_pos, mem + mem_pos, return_intermediate=True)
        
        outputs = []
        for layer_queries in intermediate_queries:
            mask_embed = self.mask_embed(layer_queries)
            masks = torch.einsum('bqc,bchw->bqhw', mask_embed, features)
            masks = F.interpolate(masks, size=input_size, mode='bilinear', align_corners=False)
            outputs.append({'class_logits': self.class_head(layer_queries), 'masks': masks})
        
        final = outputs[-1]
        final['aux_outputs'] = outputs[:-1]
        final['semantic'] = semantic_out
        return final
    
    @torch.no_grad()
    def match(self, outputs, targets):
        B, Q = outputs['class_logits'].shape[:2]
        class_probs = F.softmax(outputs['class_logits'], dim=-1)
        mask_probs = torch.sigmoid(outputs['masks'])
        
        indices = []
        for b in range(B):
            if len(targets[b]['labels']) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            
            gt_labels = targets[b]['labels']
            gt_masks = targets[b]['masks']
            
            cost_class = -class_probs[b][:, gt_labels]
            
            pred_flat = mask_probs[b].flatten(1)
            gt_flat = gt_masks.flatten(1)
            intersection = torch.matmul(pred_flat, gt_flat.t())
            union = pred_flat.sum(1, keepdim=True) + gt_flat.sum(1, keepdim=True).t() - intersection
            cost_dice = 1 - (2 * intersection / (union + 1e-8))
            
            cost_matrix = self.cost_class * cost_class + self.cost_dice * cost_dice
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            indices.append((torch.tensor(pred_idx, dtype=torch.long), torch.tensor(gt_idx, dtype=torch.long)))
        
        return indices
    
    def compute_loss(self, outputs, targets):
        indices = self.match(outputs, targets)
        loss = self._compute_single_loss(outputs, targets, indices)
        
        if 'aux_outputs' in outputs:
            for aux_out in outputs['aux_outputs']:
                aux_indices = self.match(aux_out, targets)
                loss += self._compute_single_loss(aux_out, targets, aux_indices)
        
        return loss
    
    def _compute_single_loss(self, outputs, targets, indices):
        B, Q, C = outputs['class_logits'].shape
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=outputs['class_logits'].device)
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]['labels'][gt_idx]
        
        loss_class = F.cross_entropy(outputs['class_logits'].flatten(0, 1), target_classes.flatten())
        
        loss_dice, loss_focal, num_masks = 0, 0, 0
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            pred_masks = outputs['masks'][b, pred_idx]
            gt_masks = targets[b]['masks'][gt_idx]
            loss_dice += dice_loss(pred_masks, gt_masks)
            loss_focal += sigmoid_focal_loss(pred_masks, gt_masks)
            num_masks += 1
        
        if num_masks > 0:
            loss_dice /= num_masks
            loss_focal /= num_masks
        
        return self.w_class * loss_class + self.w_dice * loss_dice + self.w_focal * loss_focal
    
    def postprocess(self, output, conf_threshold=0.5, mask_threshold=0.5, mode='semantic'):
        if mode == 'semantic':
            return self._postprocess_semantic(output, conf_threshold, mask_threshold)
        elif mode == 'instance':
            return self._postprocess_instance(output, conf_threshold, mask_threshold)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _postprocess_semantic(self, output, conf_threshold=0.5, mask_threshold=0.5):
        class_probs = F.softmax(output['class_logits'], dim=-1)
        mask_probs = torch.sigmoid(output['masks'])
        
        B, Q, _ = class_probs.shape
        _, _, H, W = mask_probs.shape
        
        semantic_masks = []
        for b in range(B):
            pred_classes = torch.argmax(class_probs[b], dim=-1)
            max_probs = torch.max(class_probs[b], dim=-1)[0]
            
            no_object_idx = class_probs.shape[-1] - 1
            valid = (pred_classes != no_object_idx) & (max_probs > conf_threshold)
            
            semantic_mask = torch.zeros((H, W), dtype=torch.long, device=class_probs.device)
            for q in range(Q):
                if valid[q]:
                    mask = mask_probs[b, q] > mask_threshold
                    semantic_mask[mask] = pred_classes[q] + 1
            semantic_masks.append(semantic_mask)
        
        return torch.stack(semantic_masks)
    
    def _postprocess_instance(self, output, conf_threshold=0.5, mask_threshold=0.5, min_mask_area=100):
        class_probs = F.softmax(output['class_logits'], dim=-1)
        mask_probs = torch.sigmoid(output['masks'])
        
        B = class_probs.shape[0]
        results = []
        
        for b in range(B):
            pred_classes = torch.argmax(class_probs[b], dim=-1)
            max_probs = torch.max(class_probs[b], dim=-1)[0]
            
            no_object_idx = class_probs.shape[-1] - 1
            valid = (pred_classes != no_object_idx) & (max_probs > conf_threshold)
            
            if not valid.any():
                results.append({
                    'classes': torch.tensor([], dtype=torch.long, device=class_probs.device),
                    'masks': torch.zeros((0, *mask_probs.shape[-2:]), dtype=torch.float32, device=class_probs.device),
                    'scores': torch.tensor([], dtype=torch.float32, device=class_probs.device)
                })
                continue
            
            valid_classes = pred_classes[valid]
            valid_masks = (mask_probs[b][valid] > mask_threshold).float()
            valid_scores = max_probs[valid]
            
            mask_areas = valid_masks.sum(dim=(1, 2))
            area_valid = mask_areas >= min_mask_area
            
            if not area_valid.any():
                results.append({
                    'classes': torch.tensor([], dtype=torch.long, device=class_probs.device),
                    'masks': torch.zeros((0, *mask_probs.shape[-2:]), dtype=torch.float32, device=class_probs.device),
                    'scores': torch.tensor([], dtype=torch.float32, device=class_probs.device)
                })
                continue
            
            final_classes = valid_classes[area_valid] + 1
            final_masks = valid_masks[area_valid]
            final_scores = valid_scores[area_valid]
            
            sorted_idx = torch.argsort(final_scores, descending=True)
            
            results.append({
                'classes': final_classes[sorted_idx],
                'masks': final_masks[sorted_idx],
                'scores': final_scores[sorted_idx]
            })
        
        return results

