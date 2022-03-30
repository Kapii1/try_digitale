
# try_digitale

# Face Retrieval

## Introduction

Le but de ce projet est de créer une application web qui retournera la célébrité la plus ressemblante.
Nous avons choisi Flask comme backend et le classique trio HTML CSS JS pour le front.

## Le prémisses

Il faut bien évidemment faire quelques réglages avant tout. On va donc devoir importer plusieurs modules.

Ensuite il faut créer une "session" et un graphe pour éviter un bug par la suite.
```python
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
```
On démarre l'application par la suite avec `app = Flask(__name__)` et maintenant on configure tout ce qui touche au modèle
de l'application.

On définit quelques variables globales qui vont nous servir au long d'une session pour conserver en mémoire des adresses:
```
filename= ""
img1_representation= ""
destination = ""
```

On va aussi charger les poids du modèle et attribuer ce modèle à une variable globale qui nous servira par la suite.


## Les Routes

Il y a trois routes que nous devrons implémenter :

- /upload : qui servira à upload l'image

- /retrieval : qui calculera le retrieval

- /morph : qui calculera le morphing



### /upload

Tout d'abord nous devons dire à python que nous allons utiliser les variables globales. Le premier `if` vérifie s'il y a bien un 
fichier envoyé. Si tout est bon, on demande d'avoir tous les fichiers. On va tous les stocker dans `UPLOAD_FOLDER`. Et enfin 
on gère la réponse.

### \retrieval

Il s’agit de déterminer l'image la plus silimilaire à notre image reqûete en calculant la similarité entre cette image et les différentes images de notre base de données. 

Pour ce faire, on fait appel à la méthode de <b> similarité cosinus </b> qui permet de déterminer les ressemblences entre deux images en se basant sur le score de similarité entre leurs vecteurs caractéristiques (features).

![Capture](https://user-images.githubusercontent.com/71329302/160779029-56abfb03-eb54-4800-9a97-e851412688a8.JPG)

Le retrieval va fonctionner de la sorte :

- On conserve le path de l'image requête dans `req_image` 
- On créé une liste `L_images2` qui contiendra toutes les images de toutes les célébrités
- On charge les features précalculé en lancant `train.py`
- Ensuite on commence le traitement du modèle pour trouver la photo la plus ressemblante
- La variable `file` contiendra l'image la plus ressemblante
- Tous les appels aux modules `shutil` et `os` servent à vider les dossiers `raw_images` et `aligned_images` en prévision d'un morphing
- Ensuite on envoie la photo qui ressemble le plus dans le dossier `static`
- On retourne un json `{'path_to_file': path ,'name': Name}`

### \morphing

Le **morphing** va être lancer en utilisant une ligne de commande :

`os.system('"python ../stylegan2/align_images.py images/raw_images/ images/aligned_images/"')`

La fonction **align_images** permet de centrer et cropper le visage dans l'image de l'entrée. Elle se base sur un detecteur de landmarks pré-entraîné pour faire ceci.

Puis la deuxieme partie du morphing pour avoir le résultat final : 

`os.system('"python ../stylegan2/project_images.py ' +'images/aligned_images_B/ images/generated_images_no_tiled/ --no-tiled"')`

La fonction **project_images** permet de faire la projection et prend une liste de paramètre : 

    'src_dir', 'Directory with aligned images for projection')
    'dst_dir', 'Output directory')
    '--network-pkl''StyleGAN2 network pickle filename')
    '--vgg16-pkl''VGG16 network pickle filename')
    '--num-steps': Number of optimization steps')
    '--initial-learning-rate' 
    '--initial-noise-factor' 
    '--verbose' : 
    '--tiled', : projection en (1,512)
    '--no-tiled' :projection en (18,512)

La projection se fait en concaténant toutes les 18 sorties du réseau, ceci nous permet de gagner en détails par rapport à la projection en prenant seulement la dernière sortie.

Dans ce qui suit, nous supposons que les images réelles ont été projetées, de sorte que nous avons accès à leurs codes latents, de forme (1, 512) )ou (18, 512) selon la méthode de projection.

Et enfin on copie le fichier final et on l'envoie en réponse.

