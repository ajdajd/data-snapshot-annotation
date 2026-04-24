---
license: unknown
task_categories:
- object-detection
- image-segmentation
tags:
- pdf
- document-layout-analysis
- data-extraction
language:
- en
- fr
- es
size_categories:
- n<1K
---

# Dataset card for data-snapshot

## Dataset summary
The `data-snapshot` dataset is an annotated corpus designed for the evaluation and development of models for extracting *data snapshots* from PDF documents. A **data snapshot** is defined as a figure or table that contains quantitative data derived from statistics, indicators, or structured data sources.

## Dataset structure

The repository is organized as follows:

```
ai4data/data-snapshot/
├── annotations/<source>/per_document/*.json    # Contains annotation files per document
├── annotations/<source>/combined/*.json        # Combined annotations into 1 JSON file per source
├── documents/<source>/*.pdf                    # Raw PDFs
├── metadata/<source>/*.json                    # Document-level metadata
├── schemas/data-snapshot-eval-v1.3.schema.json # Provides the schema of the annotation file					
└── README.md
```

### Subsets
- `annotations`
  - JSON files that indicate the data snapshots: their object class (Figure / Table) and bounding box locations (in normalized `[x1, y1, x2, y2]` format, top-left origin)
  - Follows the schema provided in `data-snapshot-eval-v1.3.schema.json`
  - Provided on a per-document basis or a combined JSON file per source
- `metadata`
  - Provided on a per-document basis
- `pdf_input`
  - The original PDF document files that correspond to the annotations.

### Sources
- UNHCR
- PRWP (WIP)
- Refugee (WIP)

## Schema

The annotation files follow the **Data Snapshot Evaluation Format (v1.3)**. Below is a simplified, human-readable example of the JSON schema with explanatory comments for each field.

> **Note**: You will notice a top-level field called `predictions`. In the context of this dataset, this is a misnomer because these are actually human-labeled **annotations** (ground truth). We use the key `predictions` because we borrow this schema from the project's evaluation codebase, which uses a unified structure for both ground truth and model predictions.

```json
{
  // Canonical mapping of integer IDs to class names
  "label_map": {
    "1": "Figure",
    "2": "Table"
  },
  
  // High-level metadata about the file
  "info": {
    "schema_version": "1.3",
    "type": "ground_truth",  // Indicates these are human annotations
    "dataset_id": "data-snapshot_unhcr",
    "created_at": "2026-04-17T12:00:00Z",
    "coordinate_system": {
      "type": "normalized_xyxy",
      "range": [0.0, 1.0],  // Bounding boxes are normalized between 0 and 1
      "origin": "top_left"
    }
  },
  
  // List of documents referenced in this file
  "documents": [
    {
      "doc_id": "1_advocacy_note_mineaction_-_niger_eng.pdf",
      "doc_name": "1_advocacy_note_mineaction_-_niger_eng.pdf",
      "doc_path": "pdf_input/1_advocacy_note_mineaction_-_niger_eng.pdf"
    }
  ],
  
  // Per-page container of objects; these contain the ground truth annotations
  "predictions": [
    {
      "page_id": "1_advocacy_note_mineaction_-_niger_eng.pdf::p001",
      "doc_id": "1_advocacy_note_mineaction_-_niger_eng.pdf",
      "page_index": 0,  // 0-indexed page number
      // Image data for Label Studio (ignore this)
      "image": {
        "width_px": 2481,
        "height_px": 3508,
        "path": "images/1_advocacy_note_mineaction_-_niger_eng.pdf_p001.png"
      },
      "objects": [
        {
          "id": "obj_001",
          "label": "Figure",  // Matches a label_map entry
          "bbox": [0.1, 0.2, 0.8, 0.6],  // Normalized [x_min, y_min, x_max, y_max]
        }
      ]
    }
  ]
}
```

## Dataset creation
The annotations were produced through human labeling using Label Studio.

## Licensing information
[TBD]

## Citation information
[TBD]
