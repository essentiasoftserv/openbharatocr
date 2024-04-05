# openbharatocr
[![Build status](https://github.com/essentiasoftserv/openbharatocr/actions/workflows/main.yml/badge.svg)](https://github.com/essentiasoftserv/openbharatocr/actions/workflows/main.yml)

openbharatocr is a Python library developed as open-source, designed specifically for optical character recognition (OCR) of Indian government documents.

The features of this package:
- It offers comprehensive support for the majority of Indian government documents, covering a wide range of document types. 


#### Installation


```
    pip install openbharatocr
```


**Pan Card**

This function takes the path of a PAN card image as input and returns its information in the form of a dictionary.

```
    import openbharatocr 
    dict_output = openbharatocr.pan(image_path)
```


**Aadhaar Card**

The two functions accepts the file paths of the front and back images of an Aadhaar card as input and returns their corresponding information encapsulated in a dictionary.

```
    import openbharatocr 
    dict_output = openbharatocr.front_aadhaar(image_path)
    dict_output = openbharatocr.back_aadhaar(image_path)
```

**Driving Licence**

This function takes the path of a Driving Licence card image as input and returns its information in the form of a dictionary.

```
    import openbharatocr 
    dict_output = openbharatocr.driving_licence(image_path)
```

**Passport**

This function takes the path of a Passport image as input and returns its information in the form of a dictionary.

```
    import openbharatocr 
    dict_output = openbharatocr.passport(image_path)
```

### Contribute & support
We are so pleased to your help and help you. If you wanna develop openbharatocr, Congrats! If you have problem, don't worry, create an issue here:

```
    https://github.com/essentiasoftserv/openbharatocr/issues
```

### Pre Commit
Note: Before committing your changes, run pre-commits 

```
    pre-commit run --all
```
