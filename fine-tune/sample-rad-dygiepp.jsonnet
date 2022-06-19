local template = import "template.libsonnet";

template.DyGIE {
    bert_model: "../model/ckpt-seg-note",
  data_paths: {
    train: "../data/JSON_DATA/radgraph/train.jsonl",
    validation: "../data/JSON_DATA/radgraph/dev.jsonl",
    test: "../data/JSON_DATA/radgraph/test.jsonl",
  },
  cuda_device: 0,
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
	target_task: "relation",
  trainer +: {
    num_epochs: 30,
    checkpointer +: {
        num_serialized_models_to_keep: 1,
      },
  },
}
