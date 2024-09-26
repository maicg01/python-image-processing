model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth}).to('cuda')
model.load_state_dict(torch.load(save_model_path))

num_images = 10

fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

val_dataset = NYU(val_paths, mode='val') 
model.eval()
for i in range(num_images):
    sample = val_dataset[i]
    img, depth = sample['image'], sample['depth']
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
   
    with torch.inference_mode():
        pred = model(img.unsqueeze(0).to('cuda'))
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
    img = img*std + mean
     
    axes[i, 0].imshow(img.permute(1,2,0))
    axes[i, 0].set_title('Image')
    axes[i, 0].axis('off')

    max_depth = max(depth.max(), pred.cpu().max())
    
    im1 = axes[i, 1].imshow(depth, cmap='viridis', vmin=0, vmax=max_depth)
    axes[i, 1].set_title('True Depth')
    axes[i, 1].axis('off')
    fig.colorbar(im1, ax=axes[i, 1])
    
    im2 = axes[i, 2].imshow(pred.cpu(), cmap='viridis', vmin=0, vmax=max_depth)
    axes[i, 2].set_title('Predicted Depth')
    axes[i, 2].axis('off')
    fig.colorbar(im2, ax=axes[i, 2])

plt.tight_layout()
