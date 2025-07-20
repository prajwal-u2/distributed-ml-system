namespace py computenode

struct Weights {
  1: list<list<double>> V,
  2: list<list<double>> W,
}

struct TrainResult {

  1: bool accepted,

  2: double training_error,

  3: Weights weights
}

service compute_node {

  TrainResult train_with_weights(
    1: string training_file,
    2: Weights initial_weights,
    3: double load_probability,
    4: i32 k,
    5: i32 h,
    6: double eta,
    7: i32 epochs,
    8: i32 scheduling_policy
  )
}
