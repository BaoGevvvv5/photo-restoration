python imagecreate.py --input_dirimg /home/baoge/imagemask/input --output_dirmask /home/baoge/imagemask/mask --output_dirmasked /home/baoge/imagemask/imgmasked --HEIGHT 32 --WIDTH 32

python /home/baoge/edgeconnect/project/test.py --checkpoints /home/baoge/edgeconnect/project/checkpoints/places2 --input /home/baoge/imagemask/imgmasked --mask /home/baoge/imagemask/mask --output /home/baoge/imagemask/output1

python /home/baoge/generative_inpainting-master/metrics_part.py --data-path /home/baoge/imagemask/input --output-path /home/baoge/imagemask/output1 --mask-path /home/baoge/imagemask/mask
