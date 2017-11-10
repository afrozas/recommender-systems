from helpers.load_file import DatasetLoader

def run():
	"""
	"""
	dl = DatasetLoader()
	dl.generate_dicts()
	ratings = dl.ratings

	# add code to run rec sys one by one followed by tests

if __name__ == '__main__':
	run()