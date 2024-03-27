# openbharatocr
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

**Driving Licence**

In this function, Driving Licence image will pass as path in the function and the output will be in dict.

```
    import openbharatocr 
    dict_output = openbharatocr.driving_licence(image_path)
```

### Contribute & support
We are so pleased to your help and help you, If you wanna develop openbharatocr, Congrats or if you have problem, don't worry create an issue here:

```
    https://github.com/essentiasoftserv/openbharatocr/issues
```

### Pre Commit
Note: Before commit your changes, run pre-commits 

```
    pre-commit run --all
```
