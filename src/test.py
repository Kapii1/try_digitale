import os
from time import time
from pathlib import Path
import glob
import os

def clean_up_temporary_folder(temporary_input_folder,
                              output_folder,
                              file_extension='.png',
                              verbose=False):

  file_names = glob.glob(output_folder + '*' + file_extension)

  for f in file_names:
    f_path = Path(f)
    base_name = f_path.name

    g = temporary_input_folder + base_name

    g_path = Path(g)  
    if g_path.is_file():
      if verbose:
        print('Removing {}'.format(g))
      os.remove(g)

  return

temporary_input_folder = 'src/images/aligned_images_B/'
output_folder = 'generated_images_no_tiled/'

os.system('"cp src/images/aligned_images/*.png src/images/aligned_images_B"')
clean_up_temporary_folder(temporary_input_folder, output_folder)

t = time()
os.system('"python stylegan2/align_images.py src/images/raw_images/ src/images/aligned_images/"')
os.system('"python stylegan2/project_images.py aligned_images_B/ generated_images_no_tiled/ --no-tiled"')
print(time()-t)

