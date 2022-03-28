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

###\retrieval

Le retrieval va fonctionner de la sorte :

- On conserve le path de l'image requête dans `req_image` 
- On créé une liste `L_images2` qui contiendra toutes les images de toutes les célébrités
- On charge les features précalculé en lancant `train.py`
- Ensuite on commence le traitement du modèle pour trouver la photo la plus ressemblante
- La variable `file` contiendra l'image la plus ressemblante
- Tous les appels aux modules `shutil` et `os` servent à vider les dossiers `raw_images` et `aligned_images` en prévision d'un morphing
- Ensuite on envoie la photo qui ressemble le plus dans le dossier `static`
- On retourne un json `{'path_to_file': path ,'name': Name}`

###\morphing
Le **morphing** va être lancer en utilisant une ligne de commande :

`os.system('"python ../stylegan2/align_images.py images/raw_images/ images/aligned_images/"')`

Puis la deuxieme partie du morphing pour avoir le résultat final : 

`os.system('"python ../stylegan2/project_images.py ' +'images/aligned_images_B/ images/generated_images_no_tiled/ --no-tiled"')`

Et enfin on copie le fichier final et on l'envoie en réponse.