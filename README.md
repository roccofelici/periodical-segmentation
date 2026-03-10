### PROJECT PRESENTATION
The literary, philological and historical study of illustrated magazines encounters numerous problems, especially when it comes to nineteenth-century or early twentieth-century periodicals. The main problem is due to the scarcity of preserved editorial archives and correspondence between publishers and writers/illustrators, documents that could record the production processes and editorial practices that guided the construction of the periodicals themselves. <br>
The core idea behind this proposal is based on the observation that, if it is true that the editorial archives do not exist, it could also be true that much, we suggest, could be deduced from the periodicals themselves. For this reason, the research we wish to bring to your attention aims to study the mutation and/or persistence of layout within individual illustrated magazines issues: the recognition of recurring or discontinuous templates (i.e. “similar layouts”) could indeed provide philological data on the material history of periodicals – in particular on the evolution of the text-image relationship and the visual pervasiveness of advertising – and suggest deductions regarding editorial practices, such as the recognition of an “editorial line” instead of an “authorial intent” in the display of images. This could allow us to conjecture the type of relationship between writer(s)-publisher(s)-illustrator(s), but also to trace the presence of a more or less strong “author’s will” rather than a “publisher’s will”. In other words, the study of periodical layouts and templates could offer a sideways glance at the backstage of editorial practices. <br>
We intend to study this on a selected dataset which includes several early 20th-century Italian magazines published in Turin considered as case studies. The layout analysis is performed using a pretrained segmentation-based model which partitions images, text and captions with bounding boxes. Dimensions, position and numbering of the identified bounding boxes are then evaluated. From this data, it is computed the percentage of space occupied by images within each page. The latter will be used as one of the two main variables, together with the number of images on the page, to evaluate similarities between layout of pages. In this way, similar layouts of pages will be highlighted providing explainability of their similarity based on a quantitative aspect. Moreover, an attempt to label the clusters of similar images with specific templates will be performed.  

### Links
- [OD sharepoint](https://usi365-my.sharepoint.com/personal/pizzam_usi_ch/_layouts/15/onedrive.aspx?CT=1764095526506&OR=OWA%2DNT%2DMail&CID=61e0ce3a%2D57fd%2D180b%2D9963%2D0092f13a5648&e=5%3A2f7d7c388f494d83adff3c7d38efb360&sharingv2=true&fromShare=true&at=9&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNTExMTQwMDEuMTkiLCJPUyI6IkxpbnV4IHVuZGVmaW5lZCJ9&cidOR=Client&id=%2Fpersonal%2Fpizzam%5Fusi%5Fch%2FDocuments%2FDH%2FAPP%20ROCCO%20per%20dimensioni&FolderCTID=0x012000C74763D84F24FF449C0725411796B65E&view=0)
- [Flourish LS](https://public.flourish.studio/visualisation/27443524/)

### Corpus

PERIODICALS:  
- La donna (1905-1916) > Emeroteca Roma (1906-1910, 1913-1914, 1916) > NB digitised as single pages
- La Lettura (1901-1916) > Digiteca Braidense > downloaded 1901-1916 > NB digitised as double pages
- L'illustrazione italiana (start-1916) > BiASA (don't know how to download from there)
- L'Esposizione di Torino (1911) > Turin 1911 Project (don't know how to download from there)
- Adolescenza (start-1916) > still to be digitised
- Bianco, rosso e verde (1915-1916) > NB digitised as double pages
- Il corriere dei piccoli (1909-1916) > NB digitised as single pages
- La grande illustrazione (1914)
  
### TODO

@marta
- use [SAM3](https://github.com/facebookresearch/sam3) to correct wrong annotations @marta @rocco

@rocco
- viz annotated images on flourish
- push some data on git: examples, segmentation, sets. Leave out only big files
- git for models and images: [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage) 
- ~IMAGES BB NUMBER, IMAGES BB DIMENSION (area), IMAGES BB POSITION (baricentro?), IMAGES BB PERCENTAGE of space occupied within each page (so need to calcutare also the total area of the page, that could be ≠ from the total area of the digitisation, that often includes extra margins)~

### Notes
- [tool](https://pixspy.com/) to hover with cursor on images and check pixels coordinates.
Pixels have origin (0,0) in the top left corner and highest values on the bottom right corner.
- [reference](https://huggingface.co/spaces/atlury/yolo-document-layout-analysis/tree/main) YOLO model

### PLOT STRUCTURE
- pagina
    - journal name (to be extracted JournalName_Year_PageId)
    - year (to be extracted NomeGiornale_Anno)
    - absolute n. of images
    - percentage of space occupied by images

### To test YOLO model's API connection
Run the following command
```bash
python scripts/test_api.py data/examples/LetturaSportiva_1912_annoVIII-numeri24-25-26-27-28-pag0_resized.jpg
```

### To segment a corpus
Unzip the corpus in the data directory.
Run the following command:
```bash
python scripts/segmentation.py data/PeriodicName_Year_Id
```
e.g.:
```bash
python scripts/segmentation.py data/GrIll_1914_feb,mag_PNGdoublepage
```

### To aggregate data
e.g.:
```bash
python scripts/aggregate.py data/GrIll_1914_feb,mag_PNGdoublepage_segmentation_results.json 
```

### To create a parquet dataset
- Navigate to /notebook
- Change the name of the dataset to explore in the first code block
- _"play"_ all blocks
