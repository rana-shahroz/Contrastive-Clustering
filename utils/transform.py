import numpy as np 
import cv2
import torchvision as tv 

class GaussianBlur() : 
    
    def __init__ (self, kernel_size, min = 0.1, max=2.0) : 
        self.m = min
        self.M = max 
        self.kernel_size = kernel_size
        
    
    def __call__(self, sample) : 
        sample = np.array(sample)
        probability = np.random.random_sample()
        
        if probability < 0.5 : 
            sigma = (self.M - self.m) * np.random.random_sample() + self.m 
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        
        return sample
    
 
# As Images are not the same size for different datasets, we make them
# same size and perform our transforms generically.
class Transforms() : 
    
    def __init__(self, size, s = 1.0, mean = None, std = None, blur = False) : 
        # Transform for train images and test images
        self.train_transform = [
            tv.transforms.RandomResizedCrop(size = size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomApply([tv.transforms.ColorJitter(0.8 * s, 0.8*s, 0.8*s, 0.2*s)],
                                      p = 0.8),
            tv.transforms.RandomGrayscale(p=0.2)
        ]
        
        if blur : 
            self.train_transform.append(GaussianBlur(kernel_size=23))
        
        self.train_transform.append(tv.transforms.ToTensor())
        
        self.test_transform = [
            tv.transforms.Resize(size = (size,size)),
            tv.transforms.ToTensor(),
        ]
        if mean and std : 
            self.train_transform.append(tv.transforms.Normalize(mean = mean, std=std))
            self.test_transform.append(tv.transforms.Normalize(mean = means, std = std))
            
        
        self.train_transform = tv.transforms.Compose(self.train_transform)
        self.test_transform = tv.transforms.Compose(self.test_transform)
        
        
    def __call__(self, X) : 
        return self.train_transform(X), self.train_transform(X)