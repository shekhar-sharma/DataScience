# package.edge_direction_histogram

- Importing

  ```
  from package.edge_direction_histogram import edge_direction_histogram as edh
  ```

- Calculating edge discriptor features

  ```
  edh.edge_direction(image)
  ```

## edge_direction()

Calculate and returns 85 bins i.e is 80 local bins and 5 global bins. 

- Syntax:

  ```
  edge_direction(image)
  ```

- Parameters:

  `image`     : Image for which descriptors to be calculated. (use opencv or any other library to read image)
  
- Return type:
  
  1D Array containing 85 Features Bins


## Other Available Methods

- `getBins()`
  
  Generate all the 2x2 sub images of the image Block and return Bin array of that Block
  
  - Syntax:
  
    ```
    getBins(image_block)
    ```
  
  - Parameters:
    
    `image_block`  : Small block of the image
    
  - Return Type:
  
    1D Descriptor Array

