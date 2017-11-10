from helpers.load_file import DatasetLoader

def run():
	"""
	"""
	dl = DatasetLoader()
	dl.generate_dicts()
	ratings = dl.ratings

if __name__ == '__main__':
	run()