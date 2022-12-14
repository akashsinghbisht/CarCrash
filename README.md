# Databricks notebook source
# MAGIC %md #LibrariesImported

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md #DataframesCreated

# COMMAND ----------

#FunctiontocreateDFfromCSVFiles
def readcsvtoDF(path):
    return spark.read.option("header",True).csv(path)

#FunctionToGroupByandCountDistinctCrash_ID
def agg_crash_id_function(df):
    display(df.select(col('Crash_ID')).agg(countDistinct(col('CRASH_ID')).alias("Count")))

# COMMAND ----------

# mntPoint Common input mount point for all source paths
mntPoint="/FileStore/tables/"
charges_df=readcsvtoDF(mntPoint+"Charges_use.csv")
Damages_df=readcsvtoDF(mntPoint+"Damages_use.csv")
Endorse_df=readcsvtoDF(mntPoint+"Endorse_use.csv")
Primary_Person_df=readcsvtoDF(mntPoint+"Primary_Person_use.csv")
Restrict_use_df=readcsvtoDF(mntPoint+"Restrict_use.csv")
Units_use_df=readcsvtoDF(mntPoint+"Units_use.csv")

# COMMAND ----------

# MAGIC %md #Analysis 1: Find the number of crashes (accidents) in which number of persons killed are male?

# COMMAND ----------

# Gender = Male and Status=KILLED . counted those number of crashes as multiple person could get killed in same crash.
agg_crash_id_function(Primary_Person_df.where("PRSN_GNDR_ID='MALE' and PRSN_INJRY_SEV_ID='KILLED'"))

# COMMAND ----------

# MAGIC %md #Analysis 2: How many two wheelers are booked for crashes? 

# COMMAND ----------

# After Checking VEH_BODY_STYL_ID , Came to conclusion only option for two wheeler was MOTORCYCLE.
agg_crash_id_function(Units_use_df.where("VEH_BODY_STYL_ID='MOTORCYCLE'"))

# COMMAND ----------

# MAGIC %md #Analysis 3: Which state has highest number of accidents in which females are involved? 

# COMMAND ----------

Primary_Person_df\
.where("PRSN_GNDR_ID='FEMALE'")\
.groupBy(col('DRVR_LIC_STATE_ID'))\
.agg(countDistinct(col('CRASH_ID')).alias("Count"))\
.orderBy(col('Count').desc())\
.limit(1)\
.show()

# COMMAND ----------

# MAGIC %md #Analysis 4: Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death

# COMMAND ----------

# found out Tot_injury_cnt where dealth_cnt is present or not equal to 0 . and putting a window function to find out top 5 to 15th
Units_use_df1=Units_use_df\
.where("DEATH_CNT!=0")\
.groupBy(col("VEH_MAKE_ID"))\
.agg(sum(col("TOT_INJRY_CNT")).cast("Int").alias("TOT_INJRY_CNT"))\

windowSpec=Window.orderBy(col("TOT_INJRY_CNT").desc())

Units_use_df1.withColumn("RNK",row_number().over(windowSpec)).where("RNK Between 5 and 15").show(truncate=True)


# COMMAND ----------

# MAGIC %md #Analysis 5: For all the body styles involved in crashes, mention the top ethnic user group of each unique body style  

# COMMAND ----------

# For finding out the Vehicle Body type used Units table and for ethnicity used PrimaryPerson table .
# for counting the number of ethinicity for each body type joined table on crash_id and Unit_NBR granularity
# After Calculating for each for body type windowed on the basis of counts present for each body type
U=Units_use_df.alias("U")
P=Primary_Person_df.alias("P")
UandP=U.join(P,[U.CRASH_ID==P.CRASH_ID,U.UNIT_NBR==P.UNIT_NBR],"Inner")\
.select(U.VEH_BODY_STYL_ID,P.PRSN_ETHNICITY_ID)\
.groupBy(U.VEH_BODY_STYL_ID,P.PRSN_ETHNICITY_ID)\
.agg(count("*").alias("count_of_records"))\
# .show(truncate=False)

windowSpec1=Window.partitionBy("VEH_BODY_STYL_ID").orderBy(col("count_of_records").desc())

display(UandP.withColumn("Row_Number",row_number().over(windowSpec1)).where("Row_Number=1"))
# .show(truncate=False)

# COMMAND ----------

# MAGIC %md #Analysis 6: Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)

# COMMAND ----------

# Assumed all crashed as Cars , for alcohol took PRSN_ALC_RSLT_ID='Positive' and Contributing Factor consisting of character as ALCOHOL and DRINKING IN all 3 contributing attributes
Units_with_AlcoholASContributer=Units_use_df.where("CONTRIB_FACTR_1_ID like ('%ALCOHOL%') or CONTRIB_FACTR_1_ID like ('%DRINKING%')\
OR CONTRIB_FACTR_2_ID like ('%ALCOHOL%') or CONTRIB_FACTR_2_ID like ('%DRINKING%')\
OR CONTRIB_FACTR_P1_ID like ('%ALCOHOL%') or CONTRIB_FACTR_P1_ID like ('%DRINKING%')").select("CRASH_ID","CONTRIB_FACTR_1_ID","CONTRIB_FACTR_2_ID","CONTRIB_FACTR_P1_ID")

UNITALCOHOLCONTR=Units_with_AlcoholASContributer.alias("UNITALCOHOLCONTR")
PPDF=Primary_Person_df.alias("PPDF")
Primary_Person_df.join(Units_with_AlcoholASContributer,[PPDF.CRASH_ID==UNITALCOHOLCONTR.CRASH_ID],"INNER")\
.where("PRSN_ALC_RSLT_ID='Positive' and DRVR_ZIP is not null")\
.groupBy(col("DRVR_ZIP"))\
.agg(count("*").alias("count_records"))\
.orderBy(col("count_records").desc())\
.limit(5)\
.show()

# COMMAND ----------

# MAGIC %md #Analysis 7: Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance

# COMMAND ----------

# taken Units and kept Units into left join with damages and found out those crash_ID not present in Damage table . and made a distinct count over those
# for SCL in problem desc it was mentioned SCL~ so taken both 1 and 2 for for > 4
# For Insurance only attribute FIN_RESP_TYPE_ID was having data related to insurance so taken them all .
U1=Units_use_df.alias('U1')
D=Damages_df.alias('D')
U1.join(D,[U1.CRASH_ID==D.CRASH_ID],"LEFT")\
.where("D.Damaged_property is null AND \
split(U1.VEH_DMAG_SCL_1_ID,' ')[1]>4 AND \
split(U1.VEH_DMAG_SCL_2_ID,' ')[1]>4 AND \
FIN_RESP_TYPE_ID like '%INSURANCE%'")\
.select("U1.CRASH_ID","VEH_DMAG_SCL_1_ID","VEH_DMAG_SCL_2_ID","FIN_RESP_TYPE_ID")\
.agg(countDistinct(col('U1.CRASH_ID')).alias("Distinct_CrashID_Count"))\
.show(truncate=False)

# COMMAND ----------

# MAGIC %md #Analysis 8: Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)

# COMMAND ----------

# DBTITLE 1,Calculating state with top 25 offences
units_with_top_25_offences=Units_use_df\
.groupBy("VEH_LIC_STATE_ID")\
.agg(countDistinct("CRASH_ID").alias("total_offences"))\
.orderBy(col("total_offences").desc())\
.limit(25)\
.select("VEH_LIC_STATE_ID")\
.rdd\
.map(lambda x:x[0])\
.collect()
print(units_with_top_25_offences)

# COMMAND ----------

# DBTITLE 1,Calculating top 10 most used colors.
units_with_top_10_colors=Units_use_df\
.where("VEH_COLOR_ID!='NA'")\
.groupBy("VEH_COLOR_ID")\
.agg(count(col("CRASH_ID")).alias("total_colors_being_used"))\
.orderBy(col("total_colors_being_used").desc())\
.limit(10)\
.select("VEH_COLOR_ID")\
.rdd\
.map(lambda x:x[0])\
.collect()
print(units_with_top_10_colors)

# COMMAND ----------

# DBTITLE 1,Filtering out records with charges related to speed.
charges_with_speed=charges_df.where("lower(Charge) like '%speed%'").select("CRASH_ID","CHARGE")
display(charges_with_speed)

# COMMAND ----------

# DBTITLE 1,Final Result for 8 Problem statement
# Filtering Units with the filteredout_Conditioned dataframes
Units_with_filtered_conditions=Units_use_df\
.filter(col("VEH_COLOR_ID").isin(*units_with_top_10_colors) & col("VEH_LIC_STATE_ID").isin(*units_with_top_25_offences))\
.select("CRASH_ID","VEH_MAKE_ID","VEH_COLOR_ID","VEH_LIC_STATE_ID")

# Joining Dataframe filteredUnits and filteredCharges to find out top 5 vehicle makers with the given conditions
U3=Units_with_filtered_conditions.alias("U3")
C=charges_with_speed.alias("C")
UandC_Final=U3.join(C,[U3.CRASH_ID==C.CRASH_ID],"INNER")\
.groupBy("VEH_MAKE_ID")\
.agg(count("*").alias("Count_for_each_Veh_Makers"))\
.orderBy(col("Count_for_each_Veh_Makers").desc())\
.limit(5)\

display(UandC_Final)
