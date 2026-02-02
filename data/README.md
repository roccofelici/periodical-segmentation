NOTE: place here the folder with your corpus to be annotated

    data/Periodical_Year_Id

        - image1
        - image2 
        - ...

---

Structure of the directory **periodicals_segmentation_data**:

```
.
├── examples
│   └── LetturaSportiva_1912_annoVIII-numeri24-25-26-27-28-pag0_resized.jpg
├── parquet
│   ├── BiancoRossoVerde_1915nn1,3-1916nn1,2,3.parquet
│   ├── BiancoRossoVerde_1915nn1,3-1916nn1,2,3_annotated.parquet
│   ├── CorriereDeiPiccoli_1908-1909-1910-1913-1916.parquet
│   ├── CorriereDeiPiccoli_1908-1909-1910-1913-1916_annotated.parquet
│   ├── Donna_1906gen,giu-1907-1908-1909gen,giu-1910-1913-1914-1916.parquet
│   └── ...
├── segmentation
│   ├── CorriereDeiPiccoli_1908-1909-1910-1913-1916.csv
│   └── ...
└── sets
    ├── BiancoRossoVerde_1915nn1,3-1916nn1,2,3.csv
    ├── CorriereDeiPiccoli_1908-1909-1910-1913-1916.csv
    ├── Donna_1906gen,giu-1907-1908-1909gen,giu-1910-1913-1914-1916.csv
    ├── GrandeIllustrazione_1914_feb,mag.csv
    ├── Lettura_1901-1916.csv
    └── LetturaSportiva_1912_giu-lug.csv
```

- _examples_ comprises single images meant to be used to test the connection 
with the API of the YOLO model. This is mainly useful when running the program
script/test_api.py
- _parquet_ contains the datasets of images (annotated and non) in Parquet 
format
- _segmentation_ contains results of the first step i.e. segmenation. Whithin 
each file (one for corpus) are present the results of the annotation  of images
in terms of location of bouding boxes and labels (either text, title or image)
- _sets_ comprises the results of the aggregation: one file per corpus contains
for each page the percentage of area occupied by images, the number of images 
in the page, the periodic of reference, an id and other metadata
