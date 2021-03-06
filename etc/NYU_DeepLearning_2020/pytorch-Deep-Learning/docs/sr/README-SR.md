# Duboko uฤenje (u PyTorch-u) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Atcold/pytorch-Deep-Learning/master)

Ovaj repozitorijum ima [prateฤi sajt](https://atcold.github.io/pytorch-Deep-Learning/) gde se svi materijali kursa mogu naฤi u video i tekstualnom formatu.

<!-- English - Mandarin - Korean - Spanish - Italian - Turkish - Japanese - Arabic - French - Farsi - Russian - Vietnamese - Serbian - Portuguese - Bengali - Hungarian -->
[๐ฌ๐ง](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/README.md) &nbsp; [๐จ๐ณ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/zh/README-ZH.md) &nbsp; [๐ฐ๐ท](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/ko/README-KO.md) &nbsp; [๐ช๐ธ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/es/README-ES.md) &nbsp; [๐ฎ๐น](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/it/README-IT.md) &nbsp; [๐น๐ท](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/tr/README-TR.md) &nbsp; [๐ฏ๐ต](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/ja/README-JA.md) &nbsp; [๐ธ๐ฆ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/ar/README-AR.md) &nbsp; [๐ซ๐ท](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/fr/README-FR.md) &nbsp; [๐ฎ๐ท](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/fa/README-FA.md) &nbsp; [๐ท๐บ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/ru/README-RU.md) &nbsp; [๐ป๐ณ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/vi/README-VI.md) &nbsp; [๐ท๐ธ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/sr/README-SR.md) &nbsp; [๐ต๐น](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/pt/README-PT.md) &nbsp; [๐ง๐ฉ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/bn/README-BN.md) &nbsp; [๐ญ๐บ](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/hu/README-HU.md)


# Podeลกavanja

Da bi bilo moguฤe pratiti veลพbe, potreban je laptop sa instaliranom Miniconda-om (minimalnom verzijom Anaconda-e) i nekoliko paketa jezika Python.
Naredne instrukcije rade za Mac i Ubuntu Linux, a za Windows je potrebno da se instalira i radi u [Git BASH](https://gitforwindows.org/) terminalu.


## Podeลกavanje radnog okruลพenja

Potrebno je otiฤi na [sajt Anaconda](https://conda.io/miniconda.html).
Skinuti i instalirati *najnoviju* Miniconda verziju za *Python* 3.7 za vaลก operativni sistem.

```bash
wget <http:// link to miniconda>
sh <miniconda*.sh>
```


## Preuzimanje git repozitorijuma sa veลพbama

Kada je Miniconda spremna, potrebno je preuzeti repozitorijum kursa i nastaviti sa podeลกavanjem okruลพenja:

```bash
git clone https://github.com/Atcold/pytorch-Deep-Learning
```


## Kreirati izolovano Miniconda okruลพenje

Promeniti direktorijum (`cd`) na folder kursa, pa ukucati:

```bash
# cd pytorch-Deep-Learning
conda env create -f environment.yml
source activate pDL
```


## Startovati Jupyter Notebook ili JupyterLab

Startovati iz terminala:

```bash
jupyter lab
```

Ili, za klasiฤni interfejs:

```bash
jupyter notebook
```


## Vizuelizacija Notebook-a

*Jupyter Notebooks* se koriste kroz lekcije za interaktivnu analizu podataka i vizualizaciju.

Koristimo tamni stil i za *GitHub* i *Jupyter Notebook*.
Trebalo bi podesiti isto jer inaฤe neฤe izgledati lepo.
JupyterLab ima ugraฤenu tamnu temu koju je moguฤe odabrati, pa je potrebno izvrลกiti dodatne instalacije samo ukoliko ลพelite da koristite klasiฤan interfejs notebook-a.
Da bi se sadrลพaj video kako treba u klasiฤnom interfejsu potrebno je instalirati sledeฤe:

 - [*Jupyter Notebook* dark theme](https://userstyles.org/styles/153443/jupyter-notebook-dark);
 - [*GitHub* dark theme](https://userstyles.org/styles/37035/github-dark) i zakomentarisati `invert #fff to #181818` blok koda.
