import config
from json import load
from elasticsearch import Elasticsearch


def create_es_index(es):
    if not es.indices.exists(index=config.es_index_name):
        with open(config.es_config) as file:
            es_config = load(file)

        es.indices.create(index=config.es_index_name, body=es_config)
        print("[INFO] index " + config.es_index_name + " has been created!")
    else:
        print('[INFO] index already exists!')


def init_es():
    es = Elasticsearch(config.es_server)
    if not es.ping():
        return None
    else:
        create_es_index(es)
        return es


def get_script_query(encoding):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['face_vector']) + 0.0",
                "params": {"query_vector": encoding[0].tolist()}
            }
        }
    }
    return script_query


def es_search(es_client, script_query):
    response = es_client.search(
        index=config.es_index_name,  # name of the index
        body={
            "size": config.es_search_size,
            "query": script_query,
            "_source": {"includes": ["id"]}
        }
    )
    return response
