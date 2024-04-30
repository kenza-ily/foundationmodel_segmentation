import torch

def add_gaussian_noise(img, mean=0., std=1.):
    noise = torch.randn(img.size()) * std + mean
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

class SaltPepperTransform(torch.nn.Module):
    def forward(self, img, label, prob=0.05):
      # Create a random matrix with the same size as the image
      rand_matrix = torch.rand(img.size())

      # Create masks for salt and pepper noise
      salt_mask = (rand_matrix < prob / 2)
      pepper_mask = ((rand_matrix >= prob / 2) & (rand_matrix < prob))

      # Apply the masks to the image
      noisy_img = img.clone()  # Clone the original image to avoid modifying it directly
      noisy_img[salt_mask] = 1.0  # Set pixels in the salt mask to white
      noisy_img[pepper_mask] = 0.0  # Set pixels in the pepper mask to black

      # Do some transformations
      # new_img=add_salt_pepper_noise(img)
      
      return noisy_img, label