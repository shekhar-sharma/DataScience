# package.local_self_similarity

- Importing

```
from package.local_self_similarity import local_self_similarity as lss
```

- Calculating Local Self Similarity 

```
lss.local_self_similarity(image)
```

## local_self_similarity()

Calculate and returns the descriptors for all patch positions in the image. 

- Syntax:

  ```
  local_self_similarity(image)
  ```
  
  or
  
  ```
  local_self_similarity(image, cor_radius=40, patch_size=5, step=10)
  ```

- Parameters:

  `image`     : Image for which descriptors to be calculated. (use opencv or any other library to read image)
  
- Optional Parameters:
  
  `patch_size`    : Size of the image patch (default and preferred: 5x5). 
  
  `cor_radius`    : Radius for image region centered at patch. (default and preferred: 40)
  
  `step`      : Incremental steps (default: 10)
  
  Note: 
  
  1) Patch Size need to be odd.
  
  2) Don't change default values unless you know what you are doing.
  
- Return type:
  
  2D Descriptor Array


## Other Available Methods

- `self_similarity_descriptor()`
  
  Calculate single self-similarity descriptor for a certain patch center.
  
  - Syntax:
  
    ```
    self_similarity_descriptor(img, yp, xp, cor_radius, patch_size, radius=4, perimeter=20)
    ```
  
  - Parameters:
    
    `img`  : image
    
    `xp`  : center x-coordinate of the image patch
    
    `yp`  : center y-coordinate of the image patch
    
    `patch_size`  : Size of the image patch (preferred: 5x5). 
    
    `radius`  : Radial parts of the image region (default and preferred: 4)
    
    `parameter` : Angular divisions in the image patch (default and preferred: 4)
    
  - Return Type:
  
    1D Descriptor Array

- `patch_ssd()` : 
  
  Calculate 'sum of squares difference'
  
  - Syntax:
  
    ```
    patch_ssd(img, yp, xp, yc, xc, patch_size):
    ```
  
  - Return Type:
  
    Integer
  
