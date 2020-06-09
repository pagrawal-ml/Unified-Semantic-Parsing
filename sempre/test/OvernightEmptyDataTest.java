package com.ibm.sempre.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.sempre.Executor.Response;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.JavaExecutor;
import edu.stanford.nlp.sempre.ListValue;

public class OvernightEmptyDataTest {

	public static String inputFolder = "/Users/ashish/Documents/Tatzia/semantic_parsing/Overnight/overnightData";
	public static String outputFolder = "/Users/ashish/Documents/Tatzia/semantic_parsing/Overnight/overnightEmptyResults";
	public static String outputFolder2 = "/Users/ashish/Documents/Tatzia/semantic_parsing/Overnight/overnightNonEmptyResults";
	
	public static String inputFile = inputFolder + File.separator + "basketball.paraphrases.train.examples";
	public static String outputFile = outputFolder + File.separator + "blocks.paraphrases.train.emptyresult.examples";
	public static String outputFile2 = outputFolder2 + File.separator + "basketball.paraphrases.train.nonemptyresult.examples";
	
	private static Formula F(String s) { return Formula.fromString(s); }
	
	public static void main(String args[]) throws IOException{
		
		BufferedReader br = null;

		List<OvernightData> overnightData = new ArrayList<OvernightData>();
		br = new BufferedReader(new FileReader(inputFile));

		String line;

		boolean exampleFlag = false;
		boolean utteranceFlag = false;
		boolean originalFlag = false;
		boolean targetFormulaFlag = false;
		String utterance = null;
		String original = null;
		String targetFormula = null;
		
		while ((line = br.readLine()) != null) {
			if(line.contains("example")){
				exampleFlag = true;
			}
			
			else if(exampleFlag && line.contains("utterance")){
				String utteranceLine = line.trim().replace(")","");
				utterance = utteranceLine.substring(11);
				utteranceFlag = true;
			}else if(exampleFlag && utteranceFlag && line.contains("original")){
				String originalLine = line.trim().replace(")","");
				original = originalLine.substring(10);
				originalFlag = true;
			}else if(exampleFlag && utteranceFlag && originalFlag && line.contains("targetFormula")){
				targetFormulaFlag = true;
			}else if(exampleFlag && utteranceFlag && originalFlag && targetFormulaFlag){
				targetFormula  = line.trim();
				exampleFlag = false;
				utteranceFlag = false;
				originalFlag = false;
				targetFormulaFlag = false;
				
				OvernightData data = new OvernightData();
				data.setOriginal(original);
				data.setTargetFormula(targetFormula);
				data.setUtterance(utterance);
				overnightData.add(data);
			}
		}

		br.close();
		
		
		//Process all the Overnight queries to check for empty results queries.
		List<OvernightData> emptyResultsQueries = new ArrayList<OvernightData>();
		List<OvernightData> nonemptyResultsQueries = new ArrayList<OvernightData>();
		
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "blocks";
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "calendar";
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "housing";
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "restaurants";
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "publications";
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "socialnetwork";
		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "basketball";
//		edu.stanford.nlp.sempre.overnight.SimpleWorld.opts.domain = "recipes";

		JavaExecutor executor = new JavaExecutor();
		for(OvernightData data: overnightData){
			
			Response response = executor.execute(F(data.targetFormula), null);
			if(response.value instanceof ListValue){
				ListValue lValue = (ListValue) response.value;
				if(lValue.values.isEmpty()){
					emptyResultsQueries.add(data);
				}else{
					nonemptyResultsQueries.add(data);
				}
			}
		}
		
		/*
		FileWriter output = new FileWriter(outputFile);    
          
		for(OvernightData emptyResultsQuery: emptyResultsQueries){
			output.write(emptyResultsQuery.toString());    
		}
		
		output.close();  
		*/
		
		
		FileWriter output2 = new FileWriter(outputFile2);    
        
		for(OvernightData nonemptyResultsQuery: nonemptyResultsQueries){
			output2.write(nonemptyResultsQuery.toString());    
		}
		
		output2.close(); 
		
		System.err.println("Empty Results Queries: " + emptyResultsQueries.size());
		System.err.println("Total Queries: " + overnightData.size());
		
	}
	
}
