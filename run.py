from helpers.load_file import DatasetLoader
from helpers.errors import Error
from rec_systems.collaborative import CollaborativeFilter
from rec_systems.collaborative_baseline import CollaborativeBaselineFilter
from rec_systems.svd import SVD
from rec_systems.cur import CUR

def run():
	""" The function first loads the dataset in three forms: all_ratings, test_ratings
	    and the remaining ratings.

	    Sequentially, the four algorithms: Collaborative, Collaborative with baseline
	    approach, SVD and CUR decomposition.
 	"""
	dl = DatasetLoader()
	all_ratings = dl.generate_dicts(dl.processed_data)
	ratings = dl.generate_dicts(dl.model_data)
	test_ratings = dl.generate_dicts(dl.test_data)
	
	# add code to run rec sys one by one followed by tests
	
	# Run Colloborative Filtering
	print("\t\t\t\tRMS Error\t\tTopKPrecision\t\tPearson Error")
	predicted, true = [], []
	cf = CollaborativeFilter(ratings)
	ct = 0
	maxCount = 50 # to be modified
	for item in test_ratings:
		for user in test_ratings[item]:
			predicted_rating = cf.predict(user, item)
			true_rating = test_ratings[item][user]
			if predicted_rating > 0:
				predicted.append(predicted_rating)
				true.append(true_rating)
			ct += 1
			if(ct == maxCount): break
		if(ct == maxCount): break

	e = Error(predicted, true)
	rmsError = e.findRMSError(predicted, true)
	topKError = e.topKPrecision(100)
	spearmanError = e.spearmanError()
	print("Collaborative Filtering:     ", rmsError, "\t", topKError, "\t", spearmanError)

	# Run Colloborative Filtering with baseline approach
	predicted, true = [], []
	cfb = CollaborativeBaselineFilter(ratings, cf)
	ct = 0
	for item in test_ratings:
		for user in test_ratings[item]:
			predicted_rating = cfb.predict(user, item)
			true_rating = test_ratings[item][user]
			if predicted_rating > 0:
				predicted.append(predicted_rating)
				true.append(true_rating)
			ct += 1
			# if(ct % 20 == 0): print(ct)
			if(ct == maxCount): break
			# print(user, item, true_rating, round(predicted_rating,3))
		if(ct == maxCount): break
	
	e = Error(predicted, true)
	rmsError = e.findRMSError(predicted, true)
	topKError = e.topKPrecision(100)
	spearmanError = e.spearmanError()
	print("Collaborative with Baseline: ", rmsError, "\t", topKError, "\t", spearmanError)

	# Run SVD
	matrix = dl.generate_matrix(all_ratings)
	predicted, true = [], []
	svd = SVD(matrix)
	for i in range(svd.MAX_USERS):
		for j in range(svd.MAX_ITEMS):
			if matrix[i][j] != 0:
				predicted.append(svd.result[i,j])
				true.append(matrix[i][j])

	e = Error(predicted, true)
	rmsError = e.findRMSError(predicted, true)
	topKError = e.topKPrecision(100)
	spearmanError = e.spearmanError()
	print("SVD Decomposition: \t     ", rmsError, "\t", topKError, "\t", spearmanError)

	# Run CUR
	predicted, true = [], []
	cur = CUR(matrix)
	
	for i in range(svd.MAX_USERS):
		for j in range(svd.MAX_ITEMS):
			if matrix[i][j] != 0:
				predicted.append(cur.result[i,j])
				true.append(matrix[i][j])

	e = Error(predicted, true)
	rmsError = e.findRMSError(predicted, true)
	topKError = e.topKPrecision(100)
	spearmanError = e.spearmanError()
	print("CUR Decomposition: \t     ", rmsError, "\t", topKError, "\t", spearmanError)

if __name__ == '__main__':
	run()
