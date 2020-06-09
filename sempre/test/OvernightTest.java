package com.ibm.sempre.test;

import edu.stanford.nlp.sempre.Executor.Response;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.JavaExecutor;
import edu.stanford.nlp.sempre.NumberValue;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import edu.stanford.nlp.sempre.freebase.FreebaseValueEvaluator;
import py4j.GatewayServer;

/*
 * @author: Ashish Mittal
 */
public class OvernightTest {
	
	private static Formula F(String s) { return Formula.fromString(s); }

	  private static Value V(double x) { return new NumberValue(x); }
	  private static Value V(String x) { return Values.fromString(x); }

	 /* 
	private static String query = "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.listValue (call edu.stanford.nlp.sempre.overnight.SimpleWorld.filter "
			+ "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.getProperty "
			+ "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.singleton en.recipe) (string !type)) "
			+ "(string meal) (string =)"
			+ " (call edu.stanford.nlp.sempre.overnight.SimpleWorld.filter (call edu.stanford.nlp.sempre.overnight.SimpleWorld.getProperty "
			+ "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.singleton en.meal) (string !type)) "
			+ "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.reverse (string meal)) (string =) en.recipe.rice_pudding)))";
	*/
	
	// private static String query = "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.listValue (call edu.stanford.nlp.sempre.overnight.SimpleWorld.concat en.meal.lunch en.meal.dinner))";
	
	  private static String query = "(call edu.stanford.nlp.sempre.overnight.SimpleWorld.listValue (call edu.stanford.nlp.sempre.overnight.SimpleWorld.countSuperlative (call edu.stanford.nlp.sempre.overnight.SimpleWorld.getProperty (call edu.stanford.nlp.sempre.overnight.SimpleWorld.singleton en.meeting) (string !type)) (string max) (string attendee)))";

	  
	  public double getAccuracy(String domain, String targetQuery, String goldQuery){
		JavaExecutor executor = new JavaExecutor();
			
		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = domain;
		//Execute the target query first.
		System.out.println("Executing target query: " + targetQuery);
		
		Response targetResponse = executor.execute(F(targetQuery), null);
		System.out.println("Target Response: " + targetResponse.value);
		
		//Execute the gold query next.
		System.out.println("Executing gold query: " + goldQuery);
		Response goldResponse = executor.execute(F(goldQuery), null);
		System.out.println("GoldResponse: " + goldResponse.value);
		
		FreebaseValueEvaluator fbEvaluator = new FreebaseValueEvaluator();
		double accuracy = fbEvaluator.getCompatibility(targetResponse.value, goldResponse.value);
		
		System.out.println("Accuracy : " + accuracy);
		return accuracy; 
	  }
	  
	  /*
	  public static void main(String args[]){
		  OvernightTest app = new OvernightTest();
		    // app is now the gateway.entry_point
		  	System.out.println("Starting JVM server on port: " + 25336);
		    GatewayServer server = new GatewayServer(app, 25336);
		    server.start();
	  }
	  */
	
	  
	  public static void main(String args[]){
		  	edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "calendar";
			System.out.println("Executing Query: " + query);
			JavaExecutor executor = new JavaExecutor();
			Response response = executor.execute(F(query), null);
			
			System.out.println(response.value);
	  }
	  
	  
	  /*
	  
	  public static void main(String args[]){
		  
		  	if(args.length < 2 && args.length > 3){
		  		System.out.println("Usage 1: ");
		  		System.out.println("Usage java OvernightTest.class domain query");
		  		System.out.println("This executes the query on the given domain.");
		  		System.out.println("Usage 2: ");
		  		System.out.println("Usage java OvernightTest.class domain query goldQuery");
		  		System.out.println("This executes the both queries on the given domain and returns accuracy result.");
		  		return;
		  	}
		  	
		  	if(args.length == 2){
		  	
			  	String domain = args[0];
			  	String query = args[1];
			
			  	edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = domain;
				System.out.println("Executing Query: " + query);
				JavaExecutor executor = new JavaExecutor();
				Response response = executor.execute(F(query), null);
				
				System.out.println(response.value);
		  	}else if(args.length == 3){
		  		
		  		String domain = args[0];
			  	String targetQuery = args[1];
			  	String goldQuery = args[2];
			  	
			  	JavaExecutor executor = new JavaExecutor();
			
			  	edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = domain;
			  	//Execute the target query first.
				System.out.println("Executing target query: " + targetQuery);
				
				Response targetResponse = executor.execute(F(targetQuery), null);
				System.out.println("Target Response: " + targetResponse.value);
				
				//Execute the gold query next.
				System.out.println("Executing gold query: " + goldQuery);
				Response goldResponse = executor.execute(F(goldQuery), null);
				System.out.println("GoldResponse: " + goldResponse.value);
				
				FreebaseValueEvaluator fbEvaluator = new FreebaseValueEvaluator();
		  		double accuracy = fbEvaluator.getCompatibility(targetResponse.value, goldResponse.value);
		  		
		  		System.out.println("Accuracy : " + accuracy);
		  	}
	}
	*/
	
	
}
