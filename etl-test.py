import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek, date_format


config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''Create Spark session
        
       Parameter: None
       
       Return:
       spark: Spark session object
    
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''Process the song data files
       
       Parameters: 
       spark: Spark session object
       input_data: S3 data path for input files
       output_data: S3 data path for output files
    
    '''
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data', 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.dropDuplicates(subset=['song_id']).select(["song_id","title","artist_id","year","duration"])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table = songs_table.write.mode('overwrite').partitionBy("year","artist_id").parquet(output_data + 'songs_table/')

    # extract columns to create artists table
    artists_table = df.dropDuplicates(subset=['artist_id']).select(col("artist_id"), \
                                                                   col("artist_name").alias("name"), \
                                                                   col("artist_location").alias("location"),\
                                                                   col("artist_latitude").alias("latitude"), \
                                                                   col("artist_longitude").alias("longitude"))
    
    # write artists table to parquet files
    artists_table = artists_table.write.mode('overwrite').parquet(output_data + 'artists_table/')
    
    # create a temp view for song data
    df.createOrReplaceTempView("song_df_table")


def process_log_data(spark, input_data, output_data):
    '''Process the log data files
    
       Parameters:
       spark: Spark session object
       input_data: S3 data path for input files
       output_data: S3 data path for output files
    
    '''
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data', '*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong").cache()

    # extract columns for users table    
    users_table = df.dropDuplicates(subset=['userId']).select(col("userId").alias("user_Id"), \
                                                              col("firstName").alias("first_name"),\
                                                              col("lastName").alias("last_name"), \
                                                              col("gender"), \
                                                              col("level"))
    
    # write users table to parquet files
    users_table = users_table.write.mode('overwrite').parquet(output_data + "users_table/")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x:  datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
    df = df.withColumn("datetime", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.dropDuplicates(subset=['datetime']).select(col("datetime").alias("start_time"), \
                                                               hour(col("timestamp")).alias("hour"), \
                                                               dayofmonth(col("datetime")).alias("day"),\
                                                               weekofyear(col("datetime")).alias("week"), \
                                                               month(col("datetime")).alias("month"), \
                                                               year(col("datetime")).alias("year"),\
                                                               dayofweek(col("datetime")).alias("weekday"))
    
    # write time table to parquet files partitioned by year and month
    time_table = time_table.write.mode('overwrite').partitionBy("year","month").parquet(output_data + "time_table/")

    # read in song data to use for songplays table
    song_df = spark.sql('''
                SELECT DISTINCT song_id, artist_id, artist_name 
                FROM song_df_table
                ''')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, (df.artist == song_df.artist_name),"inner").distinct().select(df.datetime.alias("start_time"), \
                                                                                                     df.userId.alias("user_Id"),\
                                                                                                     df.level, \
                                                                                                     song_df.song_id, \
                                                                                                     song_df.artist_id, \
                                                                                                     df.sessionId.alias("session_id"), \
                                                                                                     df.location,\
                                                                                                     df.userAgent.alias("user_agent"), \
                                                                                                     year(df.datetime).alias("year"), \
                                                                                                     month(df.datetime).alias("month"))

    # write songplays table to parquet files partitioned by year and month
    songplays_table = songplays_table.write.mode('overwrite').partitionBy("year","month").parquet(output_data + "songplays/")
    
    


def main():
    ''' Main function to run process_song_data and process_log_data functions
        
        Parameters: None
    '''
    spark = create_spark_session()
    input_data = "data/"
    output_data = "output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)
    spark.stop()


if __name__ == "__main__":
    main()
