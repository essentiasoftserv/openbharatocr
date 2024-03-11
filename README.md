# bharatocr
[![Build status](https://github.com/essentiasoftserv/openbharatocr/actions/workflows/main.yml/badge.svg)](https://github.com/essentiasoftserv/openbharatocr/actions/workflows/main.yml)

openbharatocr is an opensource python library for ocr Indian government documents 

The features of this package:
- It will support mostly all the Indian government documents.  


#### Installation


```
pip install openbharatocr
```


**Pan Card**

In this function, Pan card image will pass as path in the function and the output will be in dict.

```
import openbharatocr 
dict_output = openbharatocr.pan(image_path)
```


**Aadhaar Card**

In this function, Aadhaar Card front and back image will pass as path in the function and the output will be in dict.

```
import openbharatocr 
dict_output = openbharatocr.front_aadhaar(image_path)
dict_output = openbharatocr.back_aadhaar(image_path)
```
