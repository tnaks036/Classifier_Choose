from processing_code import input_data, import_data, split_data, train_model, prediction, evaluation_model, relation_map

input_file_name = input_data()
data = import_data(input_file_name)
A = split_data(data)
B = train_model(A[0],A[1],A[2],A[3])
C = prediction(B[0],B[1],B[2],B[3],A[1],A[4])
D = evaluation_model(A[3], C[0], C[1], C[2], C[3])
print(relation_map(data))