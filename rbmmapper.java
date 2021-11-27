//Importing Java Packages
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.StringTokenizer;
import java.io.*;
import java.util.*;
//Importing Hadoop Packages
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;


public class RBMMapper extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, DoubleWritable> 
{
	/*
	 * This is the mapper of RBM training MapReduce program
	 * Note that the format of intermediate output is <IntWritable, DoubleWritable>,
	 * because the key is the number of weight (an integer), and the value is the weight's value (double)
	 */
	
	// this is the intermediate output
	private DoubleWritable weightValue = new DoubleWritable();
    private IntWritable weightPos = new IntWritable();
    
    // These are the variables and parameters used in algorithm
    private double hidbiases,epsilonvb,espilonhb,weightcost,initialmomentum,finalmomentum;
    private int numhid,numdims,numbatches,maxepoch;
    
    private Matrix hidbiases,visbiases,poshidprobs,neghidprobs,posprods,negprods,vishidinc,hdibiasinc,visbiasinc,poshidstates,data,vishid;
    private String weightline,inputData,inputNumdims,inputNumhid;
    
    public void configure(JobConf conf) {
    	/*
    	 It reads all the configurations and distributed cache from outside. 
    	 */
    	
    	// Read number of nodes in input layer and output layer from configuration 
    	inputNumdims = conf.get("numdims");
    	inputNumhid  = conf.get("numhid");
    	
    	// Read the weights from distributed cache
        Path[] pathwaysFiles = new Path[0];
        try {
               pathwaysFiles = DistributedCache.getLocalCacheFiles(conf);
               for (Path path: pathwaysFiles) {
            	   /*
            	    Reads all the distributed cache files
            	    The driver program ensures that there is only one distributed cache file 
            	    */
                   BufferedReader fis = new BufferedReader(new FileReader(path.toString()));
                   weightline = fis.readLine();
              }
         } catch (Exception e) {
                 e.printStackTrace();
         }
    }
    
    
    
    private void  initialize(){
    	/*
    	 It parses the input strings into parameters, and initialize parameters for algorithm.
    	 */
    	
    	
        epsilonW = 0.1; 
        epsilonvb = 0.1;
        espilonhb = 0.1;
        weightcost = 0.000;
        initialmomentum = 0.5;
        finalmomentum = 0.9;

        // Parse the number of nodes in input layer and output layer
        numhid = Integer.parseInt(inputNumhid);
        numdims = Integer.parseInt(inputNumdims);
        
        // Parse the weights
        String [] tokens = inputData.split("\t");
        String [] DataString;
        if (tokens.length == 1)
        // This case happens when first time read the data
        {
            DataString = tokens[0].trim().split("\\s+");
        }
        else
        // Else, the input line is output by previous layer
        {
            DataString = tokens[1].trim().split("\\s+");
        }
        
        double [] DataVector = new double[numdims];
        double [] VishidMatrix = new double[numdims * numhid];
        int count = 0;
        String line;
        String [] tempst;
    	line = weightline;
    	tempst = line.trim().split(" ");
        count = tempst.length;
        
        if (numdims != DataString.length || numdims * numhid != count)
        {
        	/*
        	 Check if the input data match the expectation  
        	 */
            throw new IllegalArgumentException("Input data and value do not match!");
        }

        for(int i = 0; i < numdims; i++)
        {
            DataVector[i] = (double)(Integer.parseInt(DataString[i]))/255.0;
        }
        for(int i = 0; i < count; i++)
        {
        	VishidMatrix[i] = Double.parseDouble(tempst[i]);
        }

        
        // Initialize the variables 
        // Most of them are matrix.
        this.data = new Matrix(DataVector,1);
        this.vishid = new Matrix(VishidMatrix, numdims); 
        // We create matrices for Hidden Bias,Visible Bias, positive phase samples, negative phase samples
        hidbiases = new Matrix(1,numhid);
        visbiases = new Matrix(1,numdims);
        poshidprobs = new Matrix(1,numhid);
        neghidprobs = new Matrix(1,numhid);
        posprods = new Matrix(numdims,numhid);
        negprods = new Matrix(numdims,numhid);
        vishidinc = new Matrix(numdims,numhid);
        hdibiasinc = new Matrix(1,numhid);
        visbiasinc = new Matrix(1,numhid);
        poshidstates = new Matrix(1,numhid);
    }     
    
  
    private void getposphase()
    {
    	/*
    	It does the positive phase of unsupervised RBM training algorithm
    	 */
    	    	
        //Start calculate the positive phase
        //Calculate the cured value of h0
        poshidprobs = data.times(vishid);
        //(1 * numdims) * (numdims * numhid)
        poshidprobs.plusEquals(hidbiases);
        //data*vishid + hidbiases
        double [] [] product_tmp2 = poshidprobs.getArray();
        int i2 = 0;
        while( i2 < numhid)
        {
                product_tmp2[0][i2] = 1/(1 + Math.exp(-product_tmp2[0][i2]));
                i2++;
        }
        posprods = data.transpose().times(poshidprobs);
        //(numdims * 1) * (1 * numhid)
        //End of the positive phase calculation, find the binary presentation of h0
        int i3 =0;
        double [] [] tmp1 = poshidprobs.getArray();
        double [] [] tmp2 = new double [1][numhid];
        Random randomgenerator = new Random();
        while (i3 < numhid)
        {
        	/*
        	 A sampling according to possiblity given by poshidprobs
        	 */
                if (tmp1[0][i3] > randomgenerator.nextDouble())
                        tmp2[0][i3] = 1;
                else tmp2[0][i3] = 0;
                i3++;
        }
        
        // poshidstates is a binary sampling according to possiblity given by poshidprobs
        poshidstates = new Matrix(tmp2);
    }
    
    private void getnegphase()
    {
    	/*
    	It does the negative phase of unsupervised RBM training algorithm
    	 */
    	
        /*start calculate the negative phase
        1. Calculate the curved value of v1,h1
        2. Find the vector of v1
        */
        Matrix negdata = poshidstates.times(vishid.transpose());
        //(1 * numhid) * (numhid * numdims) = (1 * numdims)
        negdata.plusEquals(visbiases);
        //poshidstates*vishid' + visbiases
        double [] [] tmp1 = negdata.getArray();
        int i1 = 0;
        while( i1 < numdims)
        {
                tmp1[0][i1] = 1/(1 + Math.exp(-tmp1[0][i1]));
                i1++;
        }
        
        //Find the vector of h1
        neghidprobs = negdata.times(vishid);
        //(1 * numdims) * (numdims * numhid) = (1 * numhid)
        neghidprobs.plusEquals(hidbiases);
        double [] [] tmp2 = neghidprobs.getArray();
        int i2 = 0;
        while( i2 < numhid)
        {
            tmp2[0][i2] = 1/(1 + Math.exp(-tmp2[0][i2]));
            i2++;
        }
        negprods = negdata.transpose().times(neghidprobs);
        //(numdims * 1) *(1 * numhid) = (numdims * numhid)
    }
    
    // Update the weights and biases
    // This serves as a reducer
    private void update()
    {
    	/*
    	 * It computes the update of weights using previous results and parameters
    	 */
		double momentum;
        Matrix temp1 = posprods.minus(negprods);
        Matrix temp2 = vishid.times(weightcost);
        temp1.minusEquals(temp2);
        temp1.timesEquals(hidbiases);
        
        // the final updates of weights are written in vishidinc
        vishidinc.plusEquals(temp1);
    
    }
    
    
    public void map(LongWritable key, Text value, OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException 
    {
    	/*
    	 It implements the mapper. It outputs the numbers of weight and updated weights.
    	 
    	 Note that the format of intermediate output is <IntWritable, DoubleWritable>,
         because the key is the number of weight (an integer), and the value is the weight's value (double)
    	 */
    	inputData = value.toString();
    	
    	// Go through the process
    	initialize();
    	getposphase();
    	getnegphase();
    	update();
    	
    	// Output the intermediate data 
    	// The <key, value> pairs are <weightID, weightUpdate>
    	double [][] vishidinc_array = vishidinc.getArray();
        for(int i = 0; i < numdims; i++ )
        {
            for(int j=0; j < numhid; j++ )
            {
            	weightPos.set(i * numhid + j);
            	weightValue.set(vishidinc_array[i][j]);
            	output.collect(weightPos, weightValue);
            }
        }
        
    }

 } 