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


public class RBMReducer extends MapReduceBase implements Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> 
{
	/* 
	Note that the format of intermediate data it taking is <IntWritable, DoubleWritable>, as mapper output.
	The format of final output is also <IntWritable, DoubleWritable>,because the key is the number of weight (an integer), and the value is the update of weight's value (double)
	 */
	public void reduce(IntWritable key, Iterator<DoubleWritable> values, OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException 
	{
	   double sum = 0;
	   while (values.hasNext()) 
	   {
		   //Calculate the sum of all the updates	
		   sum += values.next().get();
	   }
	   // Output the sum
	   output.collect(key, new DoubleWritable(sum));
	}
}