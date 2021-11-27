//Importing Java Packages
import java.io.IOException;
import java.util.Iterator;
//Importing Hadoop Packages
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;


public class PropReducer extends MapReduceBase implements Reducer<LongWritable, Text, LongWritable, Text> 
{
	 // It implements the reducer. It outputs the <key, value> pair directly.
	public void reduce(LongWritable key, Iterator<Text> values, OutputCollector<LongWritable, Text> output, Reporter reporter) throws IOException 
	{
       while (values.hasNext()) 
       {
   	    //Directly output the <key, value> pair
    	   output.collect(key, values.next());
       }
	}
}