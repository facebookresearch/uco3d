import json
import pandas as pd
import sqlalchemy as sa
import sqlite3

metadata = "/fsx-repligen/shared/datasets/uCO3D/dataset_export/metadata_vgg_1212_166k.sqlite"
setlists = "/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists_allcat_val1100_dec12.sqlite"

def dump_df_to_sqlite(df, output_path, table_name: str = "set_lists", table_name_length="sequence_lengths"):
    engine = sa.create_engine(f"sqlite:///{output_path}")

    res = df.to_sql(table_name, engine, index=False, if_exists="fail")

    sequence_lengths = (
        df_setlists
        .groupby(['sequence_name', 'subset'], as_index=False)
        .agg(num_frames=('frame_number', 'size'))
        .sort_values(by=['sequence_name', 'subset'])
        .reset_index(drop=True)
    )
    if categories is not None:
        sequence_lengths["category"] = sequence_lengths["sequence_name"].map(categories)
        sequence_lengths["super_category"] = sequence_lengths["sequence_name"].map(super_categories)
    sequence_lengths.to_sql(table_name_length, engine, index=False, if_exists="fail")

    with sqlite3.connect(output_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        print(cursor.fetchone())

def get_subsets(df, threshold=0.3, test_categories=["fireplug", "doughnut", "apple"]):
    is_dynamic = df['super_category'].isin(["animals_not_close_to_humans", "pets_most_friendly_to_humans", "general_animals"])
    df_static = df[~is_dynamic]
    dynamic = df[is_dynamic]

    result = {}
    dynamic_val = dynamic.sample(n=100, random_state=42)
    dynamic_train = dynamic.drop(dynamic_val.index)
    result["dynamic_train"] = dynamic_train["sequence_name"].tolist()
    result["dynamic_val"] = dynamic_val["sequence_name"].tolist()

    filtered = df_static[df_static["score"] > threshold]
    val_random = filtered.sample(n=500, random_state=42)
    val_diverse = filtered.groupby('category').sample(n=1, random_state=42)

    result["static_val_random"] = val_random["sequence_name"].tolist()
    result["static_val_diverse"] = val_diverse["sequence_name"].tolist()
    static_train_best = filtered.drop(val_random.index.union(val_diverse.index))
    result["static_train_best"] = static_train_best["sequence_name"].tolist()
    result["static_train"] = df_static.drop(val_random.index.union(val_diverse.index))["sequence_name"].tolist()

    result["debug_train"] = static_train_best[
        static_train_best["category"].isin(test_categories)
    ].groupby('category').sample(n=5, random_state=42)["sequence_name"].tolist()
    val = pd.concat([val_random, val_diverse]).drop_duplicates()
    result["debug_val"] = val[val["category"].isin(test_categories)]["sequence_name"].tolist()

    return result


def dump_subset(setlists, train_sequences, val_sequences):
    train_sequences_set = set(train_sequences)
    val_sequences_set = set(val_sequences)
    setlists_res = pd.concat((
        setlists[setlists["sequence_name"].isin(train_sequences_set)].assign(subset="train"),
        setlists[setlists["sequence_name"].isin(val_sequences_set)].assign(subset="val"),
    ))

    return setlists_res

if __name__ == "__main__":
    with sqlite3.connect(setlists) as conn:
        # SQL query to select all data from the table 'sequence_annots'
        query = "SELECT * FROM set_lists"
        # Read the data into a pandas DataFrame
        alex_setlists = pd.read_sql_query(query, conn)

    with sqlite3.connect(metadata) as conn:
        # SQL query to select all data from the table 'sequence_annots'
        query = "SELECT * FROM sequence_annots"
        # Read the data into a pandas DataFrame
        sequence_annots = pd.read_sql_query(query, conn)

    with open("/fsx-repligen/shared/datasets/uCO3D/tool/id_category.json") as f:
        categories_json = json.load(f)
    categories = {k: v['category'] for k, v in categories_json.items()}
    super_categories = {k: v['super_group'] for k, v in categories_json.items()}

    with open("/fsx-repligen/dnovotny/datasets/uCO3D/canonical_renders/v1_segmented=False/scene_to_score.json") as f:
        scores = json.load(f)

    print("Loading done")

    sequence_annots["score"] = sequence_annots["sequence_name"].map(scores).fillna(0.0)

    result = get_subsets(sequence_annots)
    alex_setlists_trainval = alex_setlists[alex_setlists["subset"] != "test"]

    print("Subsets done. Dumping")

    df_setlists = dump_subset(alex_setlists_trainval, result["debug_train"], result["debug_val"])
    dump_df_to_sqlite(df_setlists, f"/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/set_lists_3categories-debug.sqlite")

    # df_setlists = dump_subset(alex_setlists_trainval, result["static_train"], result["static_val_diverse"])
    # dump_df_to_sqlite(df_setlists, f"/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/set_lists_static_diverse_val.sqlite")

    df_setlists = dump_subset(alex_setlists_trainval, result["static_train"], result["static_val_diverse"] + result["static_val_random"])
    dump_df_to_sqlite(df_setlists, f"/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/set_lists_static-categories.sqlite")

    df_setlists = dump_subset(alex_setlists_trainval, result["dynamic_train"], result["dynamic_val"])
    dump_df_to_sqlite(df_setlists, f"/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/set_lists_dynamic-categories.sqlite")

    df_setlists = dump_subset(
        alex_setlists_trainval, result["static_train"] + result["dynamic_train"], result["static_val_diverse"] + result["static_val_random"] + result["dynamic_val"]
    )
    dump_df_to_sqlite(df_setlists, f"/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/set_lists_all-categories.sqlite")

    df_setlists = dump_subset(
        alex_setlists_trainval, result["static_train_best"], result["static_val_diverse"] + result["static_val_random"]
    )
    dump_df_to_sqlite(df_setlists, f"/fsx-repligen/shared/datasets/uCO3D/dataset_export/set_lists/set_lists_static-categories-accurate-reconstruction.sqlite")
