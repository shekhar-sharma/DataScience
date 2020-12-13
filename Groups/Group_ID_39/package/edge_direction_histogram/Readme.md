# package.edge_direction_histogram

- Importing

  ```
  from package.edge_direction_histogram import edge_direction_histogram as ed
  ```

- Calculating edge discriptor features

  ```
  ed.edge_direction(image)
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
  
  2D Array containing 85 Features Bins
