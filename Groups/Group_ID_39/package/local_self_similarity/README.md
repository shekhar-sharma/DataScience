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

Calculate aand returns the descriptors for all patch positions in the image. 

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
  
  `patch_size`    : Size of the image patch (default and preffered: 5x5). 
  
  `cor_radius`    : Radius for image region centered at patch. (default and preffered: 40)
  
  `step`      : Incremental steps (default: 10)
  
  Note: 
  
  1) Patch Size need to be odd.
  
  2) Don't change default values unless you know what you are doing.
  
- Return type:
  
  2D Array

