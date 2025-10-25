import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
from .base import BaseDetector


class ResNetFPN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.conv_out = nn.Conv2d(2048, 256, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.conv_out(x)
        return x


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


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def generalized_box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]
    
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)
    return giou


def giou_loss(boxes1, boxes2):
    giou = generalized_box_iou(boxes1, boxes2)
    return (1 - torch.diag(giou)).sum()


class DETR(BaseDetector):
    def __init__(
        self,
        num_classes=1,
        emb_dim=256,
        num_queries=100,
        nhead=8,
        enc_layers=6,
        dec_layers=6,
        pretrained=True,
        w_class=2.0,
        w_bbox=5.0,
        w_giou=2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.emb_dim = emb_dim
        self.w_class = w_class
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        
        self.cost_class = 2.0
        self.cost_bbox = 5.0
        self.cost_giou = 2.0
        
        self.backbone = ResNetFPN(pretrained=pretrained)
        
        self.pos_emb = PositionEmbedding2D(emb_dim)
        
        self.queries = nn.Embedding(num_queries, emb_dim)
        self.query_pos = nn.Embedding(num_queries, emb_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoderWithIntermediates(decoder_layer, num_layers=dec_layers)
        
        self.class_head = nn.Linear(emb_dim, num_classes + 1)
        self.bbox_head = nn.Linear(emb_dim, 4)
    
    def forward(self, x):
        B = x.shape[0]
        features = self.backbone(x)
        pos_emb = self.pos_emb(features)
        
        mem = features.flatten(2).permute(0, 2, 1)
        mem_pos = pos_emb.flatten(2).permute(0, 2, 1)
        
        mem = self.transformer_encoder(mem + mem_pos)
        
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        q_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        
        intermediate_queries = self.transformer_decoder(queries + q_pos, mem + mem_pos, return_intermediate=True)
        
        outputs = []
        for layer_queries in intermediate_queries:
            class_logits = self.class_head(layer_queries)
            bbox_pred = self.bbox_head(layer_queries).sigmoid()
            outputs.append({'class_logits': class_logits, 'bbox_pred': bbox_pred})
        
        final = outputs[-1]
        final['aux_outputs'] = outputs[:-1]
        return final
    
    @torch.no_grad()
    def match(self, outputs, targets):
        B, Q = outputs['class_logits'].shape[:2]
        log_prob = F.log_softmax(outputs['class_logits'], dim=-1)
        bbox_pred = outputs['bbox_pred']
        
        indices = []
        for b in range(B):
            if len(targets[b]['labels']) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            
            gt_labels = targets[b]['labels']
            gt_boxes = targets[b]['boxes']
            
            cost_class = -log_prob[b][:, gt_labels]
            cost_bbox = torch.cdist(bbox_pred[b], gt_boxes, p=1)
            cost_giou = -generalized_box_iou(bbox_pred[b], gt_boxes)
            
            cost_matrix = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
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
        B, Q = outputs['class_logits'].shape[:2]
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=outputs['class_logits'].device)
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]['labels'][gt_idx]
        
        empty_weight = torch.ones(self.num_classes + 1, device=outputs['class_logits'].device)
        empty_weight[-1] = 0.1
        loss_class = F.cross_entropy(outputs['class_logits'].flatten(0, 1), target_classes.flatten(), weight=empty_weight)
        
        loss_bbox, loss_giou, num_boxes = 0, 0, 0
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            pred_boxes = outputs['bbox_pred'][b, pred_idx]
            gt_boxes = targets[b]['boxes'][gt_idx]
            
            l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction='sum')
            giou_loss_val = giou_loss(pred_boxes, gt_boxes)
            
            loss_bbox += l1_loss
            loss_giou += giou_loss_val
            num_boxes += len(pred_idx)
        
        if num_boxes > 0:
            loss_bbox /= num_boxes
            loss_giou /= num_boxes
        
        return self.w_class * loss_class + self.w_bbox * loss_bbox + self.w_giou * loss_giou
    
    @torch.no_grad()
    def postprocess(self, outputs, conf_threshold=0.5):
        class_logits = outputs['class_logits']
        bbox_pred = outputs['bbox_pred']
        
        probs = F.softmax(class_logits, dim=-1)
        scores, labels = probs[:, :, :-1].max(dim=-1)
        
        results = []
        for b in range(class_logits.shape[0]):
            keep = scores[b] >= conf_threshold
            results.append({
                'boxes': bbox_pred[b][keep],
                'scores': scores[b][keep],
                'labels': labels[b][keep]
            })
        
        return results

