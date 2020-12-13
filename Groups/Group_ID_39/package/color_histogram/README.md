# package.color_histogram

- Importing
  
  ```
  from package.color_histogram import color_histogram as ch
  ```

- Calculating color discriptor features

  ```
  color = ch.ColorHistogram((8, 8, 8))
  color.Regional(image)
  ```

## Regional()

Calculate histogram of red, green and blue with masking and dividing image into 5 parts top-left, bottom-left, top-right, bottom-right and center elliple.

- Syntax:

  ```
  Regional(image)
  ```

- Parameters:

  `image`     : Image for which descriptors to be calculated. (use opencv or any other library to read image)
  
- Return type:
  
  1D Array

## Other Available Methods

- `Global()`
  
  Calculate Histogram of red, green and blue without masking
  
  - Syntax:
  
    ```
    Global(image)
    ```
  
  - Parameters:
    
    `image`     : Image for which descriptors to be calculated. (use opencv or any other library to read image)
    
  - Return Type:
  
    1D Array

