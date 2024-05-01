# Code adapted from https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from utils import get_2d_sincos_pos_embed

class MAEEncoder(nn.Module):
    def __init__(self, img_size=48, patch_size=6, in_chans=3, embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.,norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # Embed patches
        x = self.patch_embed(x)

        # Append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add pos embed
        x += self.pos_embed

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 1:, :]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Decoder(nn.Module):
    def __init__(self, input_size=48, init_ch=16):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.conv1 = ConvBlock(init_ch * 4, init_ch * 2)
        self.upconvtrans1 = nn.ConvTranspose2d(init_ch * 2, init_ch * 2, kernel_size=2, stride=2, padding=0)
        self.conv2 = ConvBlock(init_ch * 2, init_ch, kernel_size=3, stride=1, padding=1)
        self.upconvtrans2 = nn.ConvTranspose2d(init_ch, init_ch, kernel_size=2, stride=2, padding=0)
        self.final_upconvtrans = nn.ConvTranspose2d(init_ch, init_ch, kernel_size=2, stride=2, padding=0)
        self.final_conv = nn.Conv2d(init_ch, 1, kernel_size=1)  # Output single channel segmentation map

    def forward(self, x):
        x = self.conv1(x)
        x = self.upconvtrans1(x)
        x = self.conv2(x)
        x = self.upconvtrans2(x)
        x = self.final_upconvtrans(x)
        x = self.final_conv(x)
        # Resize to match input size, if necessary
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(x) # For binary segmentation

class Autoencoder(nn.Module):
    def __init__(self, img_size=48, patch_size=6, in_chans=3, embed_dim=128, depth=6, num_heads=4):
        super(Autoencoder, self).__init__()
        self.encoder = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        # Assuming encoder output needs channel adjustment
        self.transition_conv = nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=1)  # Adjust channels
        self.decoder = Decoder(img_size,embed_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        # Reshape if necessary and adjust channels
        b, n, c = encoded.shape
        h_w = int(n**0.5)
        encoded = encoded.permute(0, 2, 1).view(b, c, h_w, h_w)
        encoded = self.transition_conv(encoded)
        decoded = self.decoder(encoded)
        return decoded
    

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_type='random', mask_ratio=0.75,block_mask_ratio=0.5):
        super().__init__()

        # --------------------------------------------------------------------------
        self.mask_type = mask_type
        self.block_mask_ratio = block_mask_ratio
        self.mask_ratio = mask_ratio
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    # LAURA NEW CODE HERE
    def grid_masking(self, x, mask_ratio):
        """Parameters
            x: list of embedded patches (batch size,number patches,embeding dim of patches)
            mask ratio: % of patches to remove
            
            Output
            x_masked: patches to be kept based on grid masking (batch size, number keep patches, embeding dim of patches)
            mask: binary mask 0 is keep, 1 is remove (batch size,number original patches)   
            ids_restore: indices to restore patches to original order/ unshuffle (batch_size,number original patches) 
        """

        N, L, D = x.shape  # batch, length (num_patches), dim

        # Create an initial mask of ones (i.e indicating to remove all patches)
        mask = torch.ones([N,L], device=x.device)

        # Calculate how many patches will be kept
        len_keep = int(L * (1 - mask_ratio))
        
        # Calculate step: every how many patches will 1 be kept 
        #i.e 0.75 mask_ratio = 4 step -> keep 1/4 patches 
        step = int(L/len_keep)

        # Get indices of patches to keep and change their value to 0 (keep)
        idx_keep = torch.arange(0, L, step, device=x.device)
        mask[:,idx_keep] = 0 

        # Shuffle patches and corresponding mask, save ids to restore to correct order
        # Each patch list is shuffled differently 
        perms = torch.stack([torch.randperm(L) for _ in range(N)]) # Generate a permutation of indices
        perms = perms.to(x.device)
        mask_shuffled = torch.gather(mask, 1, perms)
        # Expand perms for gathering on a 3D tensor x
        # Note: perms.unsqueeze(-1) expands perms to [32, 64, 1] so it can broadcast across the last dimension of x
        perms_expanded = perms.unsqueeze(-1).expand(-1, -1, D)
        x_shuffled = torch.gather(x, 1, perms_expanded)
        ids_restore = torch.argsort(perms,dim=1)

        # Remove patches from x_masked that won't be kept based on binary mask 
        x_masked = []
        for i in range(x_shuffled.shape[0]):  # Loop over each batch
            valid_indices = (mask_shuffled[i]==0).nonzero(as_tuple=True)[0]  # Indices where mask is not 0
            filtered_batch = x_shuffled[i, valid_indices]  # Keep only slices with non-zero mask
            x_masked.append(filtered_batch)
        x_masked = torch.stack(x_masked)

        return x_masked, mask, ids_restore
    # LAURA NEW CODE HERE    

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def block_masking(self, x, mask_ratio):
        
        N, L, D = x.shape  # batch, length (num_patches), dim

        # Create an initial mask of ones (i.e indicating to remove all patches)
        mask = torch.ones([N,L], device=x.device)

        # Calculate how many patches will be kept
        len_keep = int(L * (1 - mask_ratio))

        # Randomly choose the starting point of the block to keep (from patch 0 to L-len_keep)
        idx_start = torch.randint(0, L - len_keep + 1, (N,), device=x.device)

        # Change the value of the block patches in the binary mask to 0 (keep)
        for i in range(N):
            mask[i, idx_start[i]:idx_start[i] + len_keep] = 0

        # Shuffle patches and corresponding mask, save ids to restore to correct order
        # Each patch list is shuffled differently 
        perms = torch.stack([torch.randperm(L) for _ in range(N)]) # Generate a permutation of indices
        perms = perms.to(x.device)
        mask_shuffled = torch.gather(mask, 1, perms)
        # Expand perms for gathering on a 3D tensor x
        # Note: perms.unsqueeze(-1) expands perms to [32, 64, 1] so it can broadcast across the last dimension of x
        perms_expanded = perms.unsqueeze(-1).expand(-1, -1, D)
        x_shuffled = torch.gather(x, 1, perms_expanded)
        ids_restore = torch.argsort(perms,dim=1)

        # Remove patches from x_masked that won't be kept based on binary mask 
        x_masked = []
        for i in range(x_shuffled.shape[0]):  # Loop over each batch
            valid_indices = (mask_shuffled[i]==0).nonzero(as_tuple=True)[0]  # Indices where mask is not 0
            filtered_batch = x_shuffled[i, valid_indices]  # Keep only slices with non-zero mask
            x_masked.append(filtered_batch)
        x_masked = torch.stack(x_masked)

        return x_masked, mask, ids_restore
    
    def semantic_masking(self, imgs, annotations):
        """
        Apply semantic masking based on annotations provided.
        imgs: input images [N, C, H, W]
        annotations: List of annotation masks [N, H, W]
        Returns masked images and mask.
        """
        device = imgs.device
        batch_size, _, height, width = imgs.shape
        mask = torch.stack(annotations).to(device)  # Ensure annotations are the same size as imgs
        
        # Resize mask to match patch size
        patch_size = self.patch_embed.patch_size[0]
        mask = F.resize(mask.unsqueeze(1), (height // patch_size, width // patch_size), interpolation=torch.nearest).squeeze(1)
        
        # Flatten mask to match the flat patches
        mask = mask.view(batch_size, -1)
        mask = (mask > 0).float()  # Ensure binary mask

        # Apply mask
        flat_imgs = self.patchify(imgs)  # Assume patchify is defined to reshape images to [N, L, patch_features]
        masked_imgs = flat_imgs * mask.unsqueeze(-1)

        return masked_imgs, mask

    def forward_encoder(self, x, mask_ratio, annotations=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Apply either random or semantic masking based on mask_type
        if self.mask_type == 'semantic' and annotations is not None:
            x, mask, ids_restore = self.semantic_masking(x, annotations)
        elif self.mask_type == 'random':
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        elif self.mask_type == 'grid':
            x, mask, ids_restore = self.grid_masking(x, self.mask_ratio)
        elif self.mask_type == 'block':
            x, mask, ids_restore = self.block_masking(x, self.block_mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def transfer_model(old_model, args):
    """
    Transfers the weights from the pre-trained model to a model for fine-tuning.

    Args:
        old_model (nn.Module): The pre-trained model.

    Returns:
        nn.Module: The model for fine-tuning model with transferred weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    pretrain_dict = old_model.state_dict()
    
    pretrain_dict_pos = {}
    fine_tune_model = Autoencoder(img_size=args.image_size, patch_size=args.patch_size,
                    in_chans=3, embed_dim=args.enc_projection_dim,
                    depth=args.enc_layers, num_heads=args.enc_num_heads)
    fine_tune_model.to(device)
    
    for key in pretrain_dict:
        # Check if the key is part of the encoder (i.e., not part of the decoder)
        if 'decoder' not in key and 'mask_token' not in key:  # Exclude decoder and mask_token related keys
            new_key = 'encoder.' + key  # Prefix with 'encoder.' to match the fine-tuning model's structure
            pretrain_dict_pos[new_key] = pretrain_dict[key]
    
    fine_tune_model.load_state_dict(pretrain_dict_pos, strict=False)
    
    return fine_tune_model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks