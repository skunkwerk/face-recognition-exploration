from PIL import Image, ImageOps

import logging

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def convert(path, new_path, resolution, depth):
	"""
	"""
	try:
		#logger.debug('converting image: ' + path)
		#logger.debug('saving to: ' + new_path)
		im = ImageOps.posterize(Image.open(path).convert('L'), depth)
		im.thumbnail((resolution, resolution))
		im.save(new_path)
	except Exception as e:
		logger.error(e, exc_info=True)