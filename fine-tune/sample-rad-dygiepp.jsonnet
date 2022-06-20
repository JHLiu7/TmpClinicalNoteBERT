local template = import "template.libsonnet";

template.DyGIE {
    bert_model: "/path/to/model/jhliu/ClinicalNoteBERT-base-uncased-NTP-MIMIC-note",
  data_paths: {
    train: "/path/to/data/JSON_DATA/radgraph/train.jsonl",
    validation: "/path/to/data/JSON_DATA/radgraph/dev.jsonl",
    test: "/path/to/data/JSON_DATA/radgraph/test.jsonl",
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
