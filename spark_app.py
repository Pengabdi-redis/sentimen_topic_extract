from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests

conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

ssc = StreamingContext(sc, 5)
ssc.checkpoint("checkpoint_TwitterApp")
dataStream = ssc.socketTextStream("127.0.0.1",9009)

def sentimen(kalimat):
    URL = "http://127.0.0.1:5000/sentimen_api/"
    PARAMS = {'kalimat':'{sentence}'.format(sentence=kalimat)}
    r = requests.get(url = URL, params = PARAMS)
    data = r.json()
    return data['sentimen']

def aggregate_tags_count(new_values, total_sum):
    return sum(new_values) + (total_sum or 0)

def get_sql_context_instance(spark_context):
    if ('sqlContextSingletonInstance' not in globals()):
        globals()['sqlContextSingletonInstance'] = SQLContext(spark_context)
    return globals()['sqlContextSingletonInstance']

def send_word_df_to_dashboard(df):
    top_words = [ t.word for t in df.select("word").collect() ]
    top_words_count = [ p.word_count for p in df.select("word_count").collect() ]
    url = 'http://localhost:5000/updateData'
    request_data = {'labels_words_data': str(top_words), 'values_count_data': str(top_words_count)}
    print(request_data)
    response = requests.post(url, data=request_data)

def send_sentimen_df_to_dashboard(sentimen_df):
    sentimen_neg = [t.sentimen_count for t in sentimen_df.filter(sentimen_df.sentimen_type == "negatif").collect()] 
    sentimen_net = [t.sentimen_count for t in sentimen_df.filter(sentimen_df.sentimen_type == "netral").collect()]
    sentimen_pos = [t.sentimen_count for t in sentimen_df.filter(sentimen_df.sentimen_type == "positif").collect()]
    url = 'http://localhost:5000/updateData'
    request_data = {'sentimen_neg_data': str(sentimen_neg[0] or 0), 'sentimen_net_data': str(sentimen_net[0] or 0),'sentimen_pos_data': str(sentimen_pos[0] or 0)}
    print(request_data)
    response = requests.post(url, data=request_data)

def process_rdd_words(time, rdd):
    print("----------- %s -----------" % str(time))
    try:
        sql_context = get_sql_context_instance(rdd.context)
        row_rdd = rdd.map(lambda w: Row(word=w[0], word_count=w[1]))
        words_df = sql_context.createDataFrame(row_rdd)
        print(words_df.show(5))
        words_df.registerTempTable("words")
        word_counts_df = sql_context.sql("select word, word_count from words order by word_count desc limit 10")
        send_word_df_to_dashboard(word_counts_df)
    except:
        e = sys.exc_info()[0]
        print("Error: %s" % e)

def process_rdd_sentimen(time, rdd):
    print("----------- %s -----------" % str(time))
    try:
        sql_context = get_sql_context_instance(rdd.context)
        row_rdd = rdd.map(lambda w: Row(sentimen_type=w[0], sentimen_count=w[1]))
        sentimen_df = sql_context.createDataFrame(row_rdd)
        print(sentimen_df.show(5))
        send_sentimen_df_to_dashboard(sentimen_df)
    except:
        e = sys.exc_info()[0]
        print("Error: %s" % e)

if __name__ == "__main__":
    words = dataStream.flatMap(lambda line: line.split(" "))
    words_to_cnt = words.map(lambda x: (x, 1))
    words_cnt = words_to_cnt.updateStateByKey(aggregate_tags_count)

    hasil_sentimen = dataStream.map(lambda x: sentimen(x))
    sentimen_to_cnt = hasil_sentimen.map(lambda x: (x, 1))
    sentimen_cnt = sentimen_to_cnt.updateStateByKey(aggregate_tags_count)

    words_cnt.foreachRDD(process_rdd_words)
    sentimen_cnt.foreachRDD(process_rdd_sentimen)

    ssc.start()
    ssc.awaitTermination()



