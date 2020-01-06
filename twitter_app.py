import settings
import tweepy
from tweepy.auth import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import mysql.connector
import requests
import socket
import json
import re
import datetime
from unidecode import unidecode
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# request to get credentials at http://developer.twitter.com
# consumer_key    = '380RHSlFcfd1MEp1piUDyQt8g'
# consumer_secret = 'AuqQofbwgLeLGxGZAj6EcTPKNqwDCHuitrcBdaXlyMPCyQUPA8'
# access_token    = '340918363-za07hjCZBMy0LGAT4Ik6KGg4VztcqaLYGGZBVYji'
# access_secret   = 'u5DqcyfoYCPOP4gjQM1l38wVCwmHU7XDUTi4zIdPfvdLn'

consumer_key='YHVQYVzSkA6z9IQkZVKRwfNpc'
consumer_secret='xGOK66tvy9HJ5gp7Ja7HtHIg0EcHesCVfQlQgtzRMvsPwWIDHj'
access_token ='932203757642727425-NlrXIvB3Chq9D3CmmeBBvjczzbZvsaE'
access_secret='ZHMfkzDLPkGl6frVCCVwEeCiBmYDrSepiUxSv176dkXfJ'

# we create this class that inherits from the StreamListener in tweepy StreamListener
class TweetsListener(StreamListener):

    def __init__(self, csocket):
        self.client_socket = csocket
    # we override the on_data() function in StreamListener
    def on_data(self, data):
        try:
            message = json.loads( data )
            # print(message)
            try:
                tweet = message['extended_tweet']['full_text']
            except:
                tweet = message['text']
            # try:
            id_str = message["id_str"]
            # except:
                # id_str = None
            # try:
            # created_at = message["created_at"]
            created_at = datetime.datetime.strptime(message["created_at"],'%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d %H:%M:%S')
            # except
                # created_at = None
            tweet_demojify = deEmojify(tweet)
            tweet_clean = stopword.remove(clean_tweet(tweet_demojify))
            tweet_sentiment = sentimen(tweet_clean)
            # try:
            # user_created_at = message["user"]["created_at"]
            user_created_at = datetime.datetime.strptime(message["user"]["created_at"],'%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d %H:%M:%S')
            # except:
                # user_created_at = None
            # try:
            user_location = deEmojify(message["user"]["location"])
            # except:
                # user_location = None
            # try:
            user_description = deEmojify(message["user"]["description"])
            # except:
                # user_description = None
            # Store all data in MySQL
            if mydb.is_connected():
                mycursor = mydb.cursor()
                sql = "INSERT INTO {} (id_str, created_at, raw_tweet, clean_tweet, sentimen_tweet, user_created_at, user_location, user_description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)".format(settings.TABLE_NAME)
                val = (id_str, created_at, unidecode(tweet), tweet_clean, tweet_sentiment, user_created_at, user_location,                 user_description)
                mycursor.execute(sql, val)
                mydb.commit()
                mycursor.close()
            tweet_data = bytes(unidecode(tweet_clean) + "\n", 'utf-8')
            print( tweet_data )
            self.client_socket.send( tweet_data )
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def if_error(self, status):
        print(status)
        return True

def sentimen(kalimat):
    URL = "http://127.0.0.1:5000/sentimen_api/"
    PARAMS = {'kalimat':'{sentence}'.format(sentence=kalimat)}
    r = requests.get(url = URL, params = PARAMS)
    data = r.json()
    return data['sentimen']

def clean_tweet(tweet): 
    ''' 
    Use simple regex statements to clean tweet text by removing links and special characters
    '''
    tweet_remRT = tweet.replace("RT","")
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet_remRT).split()) 

def deEmojify(text):
    '''
    Strip all non-ASCII characters to remove emoji characters
    '''
    if text:
        return text.encode('ascii', 'ignore').decode('ascii')
    else:
        return None

def send_tweets(c_socket):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=['ahok','jokowi']) # this is the topic we are interested in

if __name__ == "__main__":
    mydb = mysql.connector.connect(
        host="localhost",
        port="3306",
        #unix_socket = 'localhost:/Applications/MAMP/tmp/mysql/mysql.sock',
        user="root",
        passwd="YUrio_123",
        database="twitterdb",
        charset = 'utf8'
    )

    if mydb.is_connected():
        '''
        Check if this table exits. If not, then create a new one.
        '''
        mycursor = mydb.cursor()
        mycursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{0}'
            """.format(settings.TABLE_NAME))
        if mycursor.fetchone()[0] != 1:
            mycursor.execute("CREATE TABLE {} ({})".format(settings.TABLE_NAME, settings.TABLE_ATTRIBUTES))
            mydb.commit()
        mycursor.close()

    new_skt = socket.socket()         # initiate a socket object
    host = "127.0.0.1"     # local machine address
    #port = 5555                 # specific port for your service.
    port = 9009
    new_skt.bind((host, port))        # Binding host and port

    print("Now listening on port: %s" % str(port))

    new_skt.listen(5)                 #  waiting for client connection.
    c, addr = new_skt.accept()        # Establish connection with client. it returns first a socket object,c, and the address bound to the socket

    print("Received request from: " + str(addr))
    print("this is c " + str(c) + "\n")
    # and after accepting the connection, we can send the tweets through the socket
    send_tweets(c)

