import torch
import torch.nn as nn

class InterpolationLoss(nn.Module):
    def __init__(self, lambda_interpolation=1.0):
        super(InterpolationLoss, self).__init__()
        self.lambda_interpolation = lambda_interpolation
        self.loss_fn = nn.MSELoss()

    def forward(self, interpolated_images, real_images):
        loss = 0.0
        for img in interpolated_images:
            loss += self.loss_fn(img, real_images)  # Compare interpolated image with real image
        return self.lambda_interpolation * loss

def interpolate_latent_space(model, z1, z2, num_steps=10):
    # Perform linear interpolation between two latent vectors z1 and z2
    interpolation_steps = torch.linspace(0, 1, num_steps)
    interpolated_images = []
    for alpha in interpolation_steps:
        z_interpolated = alpha * z1 + (1 - alpha) * z2  # Interpolate latent vectors
        generated_image = model.decode(z_interpolated)  # Decode to generate image
        interpolated_images.append(generated_image)
    return interpolated_images
    
def compute_loss(model, data, interpolator):
    # 获取健康和疾病图像
    health_image = data['health']
    disease_image = data['disease']
    
    # 从潜在空间中选择两个健康图像和两个疾病图像
    z_health_1, z_health_2 = random_latent_vectors(health_image)  # 从健康图像中随机选择两个潜在向量
    z_disease_1, z_disease_2 = random_latent_vectors(disease_image)  # 从疾病图像中随机选择两个潜在向量
    
    # 生成健康到疾病和疾病到健康的插值图像
    health_to_disease_images = interpolate_latent_space(model, z_health_1, z_disease_1)
    disease_to_health_images = interpolate_latent_space(model, z_disease_1, z_health_1)
    
    # 计算插值损失
    health_to_disease_loss = interpolator(health_to_disease_images, disease_image)
    disease_to_health_loss = interpolator(disease_to_health_images, health_image)
    
    # 计算模型的标准损失（例如生成健康到疾病的损失和生成疾病到健康的损失）
    standard_loss = model.compute_loss(data)
    
    # 总损失 = 标准损失 + 插值损失
    total_loss = standard_loss + health_to_disease_loss + disease_to_health_loss
    return total_loss

total_loss = compute_loss(model, data, interpolator)
total_loss.backward()
optimizer.step()
