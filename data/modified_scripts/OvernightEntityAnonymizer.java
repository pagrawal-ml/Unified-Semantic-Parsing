//package overnight;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

// Parag: Modified it to run for all the domains at once.

public class OvernightEntityAnonymizer {

	static Boolean anonymizeNumbers = false;
	
	//static String domain = "calendar";
	
	/*
	
	static String sourceFolder = "/home/karnad/pCloudDrive/acl code/overnight-entity-anonymization-master/resources/pruned_new";
	
	static String sourceTrainFile = sourceFolder + File.separator + domain + "_train.tsv.prune";
	static String sourceTestFile = sourceFolder + File.separator + domain + "_test.tsv.prune";
	
	static String destFolder = "/home/karnad/pCloudDrive/acl code/overnight-entity-anonymization-master/resources/java_out";

	static String outputTrainFile = destFolder + File.separator + domain + "_train.tsv.entity.prune.txt";
	static String outputTestFile = destFolder + File.separator + domain + "_test.tsv.entity.prune.txt";
	
	static String outputTrainTransformationFile = destFolder + File.separator + domain + "_train.trans.txt";
	static String outputTestTransformationFile = destFolder + File.separator + domain + "_test.trans.txt";
	*/
	
	public static String REGEX_FIND_WORD=".*\\b%s\\b.*";

	public static boolean containsWord(String text, String word) {
	    String regex=String.format(REGEX_FIND_WORD, Pattern.quote(word));
	    return text.matches(regex);
	}
	
	public static void transform(String domain, String sourceFile, String outputFile, 
			String transformationFile, Map<String, String> nlToEntityMap) throws IOException{
		
		//Read source train file.
		System.out.println("Reading file: " + sourceFile);
		File train_file = new File(sourceFile);
        BufferedReader br = new BufferedReader(new FileReader(train_file));

        String readLine = "";
        
      //Sort the keys from longest to shortest based on length ("thai cafe" should be matched before "thai")
        Set<String> unsortedKeys = nlToEntityMap.keySet();
        String[] keys = new String[unsortedKeys.size()];
        keys = (String[]) unsortedKeys.toArray(keys);
        Arrays.sort(keys,new Comparator<String>()
        {
       	  public int compare(String s1,String s2)
       	   {
       	    return s2.length() - s1.length();
       	    }
        }); 

        System.out.println("Writing to file: " + outputFile);
        FileWriter outputWriter = new FileWriter(outputFile);
        FileWriter transformationWriter = new FileWriter(transformationFile);
        while ((readLine = br.readLine()) != null) {
        
        	 String parts[] = readLine.split("\\\t");
             if(parts.length != 2){
            	 System.err.println("Bad Line " + readLine);
            	 continue;
             }
             
             String nlQuery = parts[0];
             
             //Preprocess nlQuery
             if(domain.equals("recipes")){
            	 //Typos
            	 /*
            	 nlQuery = nlQuery.replaceAll("receipe", "recipe");
            	 nlQuery = nlQuery.replaceAll("quice", "quiche");
            	 nlQuery = nlQuery.replaceAll("ingrediant", "ingredient");
            	 nlQuery = nlQuery.replaceAll("ingedients", "ingredients");
            	 nlQuery = nlQuery.replaceAll("preperation", "preparation");
            	 nlQuery = nlQuery.replaceAll("in2004", "in 2004");
            	 */
             }else if(domain.equals("restaurants")){
            	 //Entity Normalizations
            	 /*
            	 nlQuery = nlQuery.replaceAll("3 or 2 menu", "3 menu or 2 menu");
            	 nlQuery = nlQuery.replaceAll("a 3 or a 5 star", "a 3 star or a 5 star");
            	 nlQuery = nlQuery.replaceAll("3 or 5 star", "3 star or 5 star");
            	 nlQuery = nlQuery.replaceAll("3 to 5 stars", "3 stars to 5 stars");
            	 nlQuery = nlQuery.replaceAll("2 or 3 dollar", "2 dollar or 3 dollar");
            	 nlQuery = nlQuery.replaceAll("30 or 40 reviews", "30 reviews or 40 reviews");
            	 */
            	 nlQuery = nlQuery.replaceAll("threestar", "3 star");
            	 
            	 //Typos
            	 /*
            	 nlQuery = nlQuery.replaceAll("resturant", "restaurant");
				 nlQuery = nlQuery.replaceAll("neighbourhood", "neighborhood");
				 nlQuery = nlQuery.replaceAll("2dollarsign", "2 dollar sign");
				 nlQuery = nlQuery.replaceAll("thia", "thai");
            	 */
             }else if(domain.equals("publications")){
            	 
            	 if(anonymizeNumbers){
             		//number transformation rules.
            		 nlQuery = nlQuery.replaceAll("two or fewer articles","two articles or fewer");
            		 nlQuery = nlQuery.replaceAll("two or more articles","two articles or more");
            		 nlQuery = nlQuery.replaceAll("two or less articles","two articles or less");
            		 nlQuery = nlQuery.replaceAll("two or fewer citations","two citations or fewer");
            		 nlQuery = nlQuery.replaceAll("two or fewer references","two references or fewer");

            	 }
            	 
            	 //Typos
            	 /*
            	 nlQuery = nlQuery.replaceAll("bymultivariate", "by multivariate");
            	 nlQuery = nlQuery.replaceAll("citesmultivariate","cites multivariate");
            	 nlQuery = nlQuery.replaceAll("anals","annals");
            	 nlQuery = nlQuery.replaceAll("refernces","references");
            	 nlQuery = nlQuery.replaceAll("publised","published");
            	 nlQuery = nlQuery.replaceAll("citys","cites");
            	 nlQuery = nlQuery.replaceAll("articiles","articles");
            	 */
             }else if(domain.equals("housing")){
            	 
            	 //Typos
            	 /*
            	 nlQuery = nlQuery.replaceAll("unis","units");
            	 nlQuery = nlQuery.replaceAll("sesamre","sesame");
            	 nlQuery = nlQuery.replaceAll("medtown","midtown");
            	 nlQuery = nlQuery.replaceAll("condomonium","condominium");
            	 */
             }else if(domain.equals("calendar")){
            	 
            	 nlQuery = nlQuery.replaceAll("jan 2 or 3","jan 2 or jan 3");
            	 nlQuery = nlQuery.replaceAll("jan 2 or 3rd	","jan 2 or jan 3");
            	 nlQuery = nlQuery.replaceAll("january 2nd or 3rd","january 2nd or january 3rd");
            	 
            	 if(anonymizeNumbers){
            		//number transformation rules.
            		 nlQuery = nlQuery.replaceAll("one or three hours","one hour or three hours");
            		 nlQuery = nlQuery.replaceAll("three or one hours","three hours or one hours");
            		 nlQuery = nlQuery.replaceAll("one to three hours","one hour to three hours");
            		 nlQuery = nlQuery.replaceAll("1 and 3 hours","1 hour and 3 hours");
            	 }
             }else if(domain.equals("blocks")){
            	 
            	 nlQuery = nlQuery.replaceAll("block 1 or 2","block 1 or block 2");
            	 nlQuery = nlQuery.replaceAll("brick 1 or 2","brick 1 or brick 2");
            	 nlQuery = nlQuery.replaceAll("blocks 1 or 2","block 1 or block 2");
            	 nlQuery = nlQuery.replaceAll("bricks 1 or 2","brick 1 or brick 2");
            	 
             }
             
             String parse = parts[1];
             
             String processedNLQuery = nlQuery;
             String processedParse = parse;
             
             Map<String, String> transformation = new HashMap<String, String>();
             
                         
             
             for(String key: keys){
            	 if(containsWord(processedNLQuery, key)){
            		 processedNLQuery = processedNLQuery.replace(key, "entity_" + key.replace(" ", "_"));
            		 
            		 String val = nlToEntityMap.get(key);
            		 processedParse = processedParse.replace(val, "entity_" + val.replace(" ", "_"));
            		 
            		 transformation.put("entity_"+  key.replace(" ", "_"), val.replace(" ", "_"));
            	 }
             }
             
             Map<String, String> processedTransformation = new HashMap<String, String>();

             String entityType = "e";
             
             int index = 0;
             int dateIndex = 0;
             int numIndex = 0;
             String queryWords[] = processedNLQuery.split(" ");
             for(String queryWord: queryWords){
            	 if(queryWord.startsWith("entity_")){
            		 
            		 String transformedValue = transformation.get(queryWord);
            		 if(transformedValue == null){
            			 System.out.println(readLine);
            			 System.out.println(transformation);
            		 }
            		 if(transformedValue.contains("date") || transformedValue.contains("time")){
            			 entityType = "d";
            		 }else if(transformedValue.contains("number")){
            			 entityType = "n";
            		 }else{
            			 entityType = "e";
            		 }
  
            		if(entityType.equals("e")){
	            		processedNLQuery = processedNLQuery.replace(queryWord, entityType + index);
	            		processedTransformation.put(entityType  + index, transformedValue);
            		}else if(entityType.equals("n")){
	            		processedNLQuery = processedNLQuery.replace(queryWord, entityType + numIndex);
	            		processedTransformation.put(entityType  + numIndex, transformedValue);
            		}else{
            			processedNLQuery = processedNLQuery.replace(queryWord, entityType + dateIndex);
	            		processedTransformation.put(entityType  + dateIndex, transformedValue);
            		}
            		
            		String parseWords[] = processedParse.split(" ");
                    for(String parseWord: parseWords){
                    	if(parseWord.startsWith("entity_"+transformedValue)){
	                   		 if(entityType.equals("e")){
	       	            		 processedParse = processedParse.replace(parseWord, entityType + index);
	       	            		 index += 1;
	                   		 }if(entityType.equals("n")){
	       	            		 processedParse = processedParse.replace(parseWord, entityType + numIndex);
	       	            		 numIndex += 1;
	                   		 }else{
	                   			 processedParse = processedParse.replace(parseWord, entityType + dateIndex);
	       	            		 dateIndex += 1;
	                   		 }
                   	 	}
                    }
            		
            	 }
             }
             
             processedNLQuery = processedNLQuery.trim();
             processedParse = processedParse.trim();
             
             outputWriter.write(processedNLQuery + "\t" + processedParse + "\n"); 
             transformationWriter.write(processedTransformation.toString() + "\n");

         }
         br.close();
         outputWriter.close();
         transformationWriter.close();
	}
	
	public static Map<String, String> getRecipesNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();
		
		nlToEntityMap.put("rice pudding", "en.recipe.rice_pudding");
		nlToEntityMap.put("rice puddings", "en.recipe.rice_pudding");
		nlToEntityMap.put("lunch", "en.meal.lunch");
		nlToEntityMap.put("milk", "en.ingredient.milk");
		nlToEntityMap.put("2004", "date:2004:-1:-1");
		nlToEntityMap.put("2010", "date:2010:-1:-1");
		nlToEntityMap.put("quiche", "en.recipe.quiche");
		nlToEntityMap.put("spinach", "en.ingredient.spinach");
		nlToEntityMap.put("dinner", "en.meal.dinner");
		
		return nlToEntityMap;
	}
	
	public static Map<String, String> getPublicationsNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();

		nlToEntityMap.put("multivariate data analysis", "en.article.multivariate_data_analysis");
		nlToEntityMap.put("efron", "en.person.efron");
		nlToEntityMap.put("lakoff", "en.person.lakoff");
		nlToEntityMap.put("computational linguistics", "en.venue.computational_linguistics");
		nlToEntityMap.put("statistics", "en.venue.annals_of_statistics");
		nlToEntityMap.put("2004", "date:2004:-1:-1");
		nlToEntityMap.put("2010", "date:2010:-1:-1");
		
		if(anonymizeNumbers){
			nlToEntityMap.put("two articles", "number2 entity-en.article");
			nlToEntityMap.put("two article", "number2 entity-en.article");
			nlToEntityMap.put("two other articles", "number2 entity-en.article");
			nlToEntityMap.put("2 other articles", "number2 entity-en.article");
			nlToEntityMap.put("two references", "number2 entity-en.article");
			nlToEntityMap.put("two citations", "number2 entity-en.article");
			//nlToEntityMap.put("single article", "number2 entity-en.article");
			
		}
		
		return nlToEntityMap;
	}
	
	public static Map<String, String> getBasketballNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();

		nlToEntityMap.put("kobe bryant", "en.player.kobe_bryant");
		nlToEntityMap.put("kobebryant", "en.player.kobe_bryant");
		nlToEntityMap.put("kobe bryants", "en.player.kobe_bryant");
		nlToEntityMap.put("kob bryant", "en.player.kobe_bryant");
		nlToEntityMap.put("kobe", "en.player.kobe_bryant");
		nlToEntityMap.put("bryants", "en.player.kobe_bryant");
		nlToEntityMap.put("lebron james", "en.player.lebron_james");
		nlToEntityMap.put("lakers", "en.team.lakers");
		nlToEntityMap.put("la laker", "en.team.lakers");
		nlToEntityMap.put("los angeles lakers", "en.team.lakers");
		nlToEntityMap.put("cleveland cavaliers", "en.team.cavaliers");
		nlToEntityMap.put("cavaliers", "en.team.cavaliers");
		nlToEntityMap.put("point guard", "en.position.point_guard");
		nlToEntityMap.put("forward", "en.position.forward");
		nlToEntityMap.put("2004", "date:2004:-1:-1");
		nlToEntityMap.put("2010", "date:2010:-1:-1");
		
		
		return nlToEntityMap;
	}
	
	public static Map<String, String> getBlocksNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();

		nlToEntityMap.put("block 1", "en.block.block1");
		nlToEntityMap.put("block 2", "en.block.block2");
		nlToEntityMap.put("brick 1", "en.block.block1");
		nlToEntityMap.put("brick 2", "en.block.block2");
		nlToEntityMap.put("pyramid", "en.shape.pyramid");
		nlToEntityMap.put("pyramidshaped", "en.shape.pyramid");
		nlToEntityMap.put("cube", "en.shape.cube");
		//nlToEntityMap.put("", "en.color.red");
		//nlToEntityMap.put("", "en.color.green");
		
		return nlToEntityMap;
	}
	
	public static Map<String, String> getCalendarNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();
		
		nlToEntityMap.put("weekly standup", "en.meeting.weekly_standup");
		nlToEntityMap.put("weekly", "en.meeting.weekly_standup");
		nlToEntityMap.put("annual review", "en.meeting.annual_review");
		nlToEntityMap.put("alice", "en.person.alice");
		nlToEntityMap.put("bob", "en.person.bob");
		nlToEntityMap.put("greenberg cafe", "en.location.greenberg_cafe");
		nlToEntityMap.put("greenberg", "en.location.greenberg_cafe");
		nlToEntityMap.put("central office", "en.location.central_office");
		nlToEntityMap.put("january 2nd", "date:2015:1:2");
		nlToEntityMap.put("january second", "date:2015:1:2");
		nlToEntityMap.put("january 2", "date:2015:1:2");
		nlToEntityMap.put("jan 2", "date:2015:1:2");
		nlToEntityMap.put("jan 2nd", "date:2015:1:2");
		nlToEntityMap.put("jan 3", "date:2015:1:3");
		nlToEntityMap.put("jan 3rd", "date:2015:1:3");
		nlToEntityMap.put("january third", "date:2015:1:3");
		nlToEntityMap.put("january 3rd", "date:2015:1:3");
		nlToEntityMap.put("january 3", "date:2015:1:3");
		
		nlToEntityMap.put("10am", "( time 10 0 )");
		nlToEntityMap.put("10 am", "( time 10 0 )");
		nlToEntityMap.put("1000 am", "( time 10 0 )");
		nlToEntityMap.put("3pm", "( time 15 0 )");
		nlToEntityMap.put("3 pm", "( time 15 0 )");
		
		if(anonymizeNumbers){
			nlToEntityMap.put("3 hours", "number3 en.hour");
			nlToEntityMap.put("3 hour", "number3 en.hour");
			nlToEntityMap.put("threehour", "number3 en.hour");
			nlToEntityMap.put("1 hours", "number1 en.hour");
			nlToEntityMap.put("1 hour", "number1 en.hour");
			nlToEntityMap.put("three hours", "number3 en.hour");
			nlToEntityMap.put("three hour", "number3 en.hour");
			nlToEntityMap.put("an hour", "number1 en.hour");
			nlToEntityMap.put("one hours", "number1 en.hour");
			nlToEntityMap.put("one hour", "number1 en.hour");
		}
		
		return nlToEntityMap;
	}
	
	public static Map<String, String> getSocialNetworkNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();

		
		nlToEntityMap.put("alice", "en.person.alice");
		nlToEntityMap.put("alices", "en.person.alice");
		nlToEntityMap.put("bob", "en.person.bob");
		//nlToEntityMap.put("male", "en.gender.male");
		//nlToEntityMap.put("female", "en.gender.female");
		nlToEntityMap.put("single", "en.relationship_status.single");
		nlToEntityMap.put("singles", "en.relationship_status.single");
		nlToEntityMap.put("married", "en.relationship_status.married");
		nlToEntityMap.put("new york", "en.city.new_york");
		nlToEntityMap.put("newyork", "en.city.new_york");
		nlToEntityMap.put("beijing", "en.city.bejing");
		nlToEntityMap.put("brown university", "en.university.brown");
		nlToEntityMap.put("brown", "en.university.brown");
		//nlToEntityMap.put("", "en.university.berkeley");
		nlToEntityMap.put("ucla", "en.university.ucla");
		nlToEntityMap.put("ulca", "en.university.ucla");
		nlToEntityMap.put("computer science", "en.field.computer_science");
		//nlToEntityMap.put("", "en.field.economics");
		nlToEntityMap.put("history", "en.field.history");
		nlToEntityMap.put("google", "en.company.google");
		nlToEntityMap.put("mckinsey", "en.company.mckinsey");
		nlToEntityMap.put("mckinseys", "en.company.mckinsey");
		nlToEntityMap.put("mickinsey", "en.company.mckinsey");
		//nlToEntityMap.put("", "en.company.toyota");
		//nlToEntityMap.put("", "en.job_title.ceo");
		nlToEntityMap.put("software engineer", "en.job_title.software_engineer");
		nlToEntityMap.put("software engineers", "en.job_title.software_engineer");
		nlToEntityMap.put("program manager", "en.job_title.program_manager");
		nlToEntityMap.put("program managers", "en.job_title.program_manager");
		nlToEntityMap.put("managers", "en.job_title.program_manager");
		nlToEntityMap.put("manager", "en.job_title.program_manager");
		nlToEntityMap.put("2004", "date:2004:-1:-1");
		nlToEntityMap.put("2010", "date:2010:-1:-1");
		
		return nlToEntityMap;
	}
	
	
	public static Map<String, String> getHousingNLMap(){
		
		Map<String, String> nlToEntityMap = new HashMap<String, String>();

		nlToEntityMap.put("apartment", "en.housing.apartment");
		nlToEntityMap.put("apartments", "en.housing.apartment");
		nlToEntityMap.put("condo", "en.housing.condo");
		nlToEntityMap.put("condos", "en.housing.condo");
		nlToEntityMap.put("condominium", "en.housing.condo");
		//nlToEntityMap.put("", "en.housing.house");
		//nlToEntityMap.put("", "en.housing.flat");
		//nlToEntityMap.put("", "en.neighborhood.tribeca");
		nlToEntityMap.put("midtown west", "en.neighborhood.midtown_west");
		nlToEntityMap.put("chelsea", "en.neighborhood.chelsea");
		nlToEntityMap.put("123 sesame street", "en.housing_unit.123_sesame_street");
		nlToEntityMap.put("123 sesame st", "en.housing_unit.123_sesame_street");
		nlToEntityMap.put("123sesame street", "en.housing_unit.123_sesame_street");
		nlToEntityMap.put("900 mission avenue", "en.housing_unit.900_mission_ave"); 
		nlToEntityMap.put("900 mission ave", "en.housing_unit.900_mission_ave");
		nlToEntityMap.put("january 2", "date:2015:1:2");
		nlToEntityMap.put("february 3", "date:2015:2:3");
		
		/*
		nlToEntityMap.put("1500 dollars", "number1500 en.dollar");
		nlToEntityMap.put("1500 dollar", "number1500 en.dollar");
		nlToEntityMap.put("2000 dollars", "number2000 en.dollar");
		nlToEntityMap.put("2000 dollar", "number2000 en.dollar");
		nlToEntityMap.put("800 square feet", "number800 en.square_feet");
		nlToEntityMap.put("800 sq ft", "number800 en.square_feet");
		nlToEntityMap.put("1000 square feet", "number1000 en.square_feet");
		nlToEntityMap.put("1000 sq ft", "number1000 en.square_feet");
		*/
		
		return nlToEntityMap;
	}
	
	public static Map<String, String> getRestaurantsNLMap(){
		Map<String, String> nlToEntityMap = new HashMap<String, String>();
		
		nlToEntityMap.put("thai cafe", "en.restaurant.thai_cafe");
		nlToEntityMap.put("pizzeria juno", "en.restaurant.pizzeria_juno");
		nlToEntityMap.put("thai", "en.cuisine.thai");
		nlToEntityMap.put("italian", "en.cuisine.italian");
		nlToEntityMap.put("french", "en.cuisine.french");
		nlToEntityMap.put("midtown west", "en.neighborhood.midtown_west");
		nlToEntityMap.put("chelsea", "en.neighborhood.chelsea");
		nlToEntityMap.put("tribeca", "en.neighborhood.tribeca");
		nlToEntityMap.put("lunch", "en.food.lunch");
		nlToEntityMap.put("dinner", "en.food.dinner");
		nlToEntityMap.put("breakfast", "en.food.breakfast");
		
		
		/*
		nlToEntityMap.put("1 star", "number1 en.star");
		nlToEntityMap.put("2 star", "number2 en.star");
		nlToEntityMap.put("3 star", "number3 en.star");
		nlToEntityMap.put("4 star", "number4 en.star");
		nlToEntityMap.put("5 star", "number5 en.star");
		nlToEntityMap.put("1 stars", "number1 en.star");
		nlToEntityMap.put("2 stars", "number2 en.star");
		nlToEntityMap.put("3 stars", "number3 en.star");
		nlToEntityMap.put("4 stars", "number4 en.star");
		nlToEntityMap.put("5 stars", "number5 en.star");
		nlToEntityMap.put("1 dollar sign", "number1 en.dollar_sign");
		nlToEntityMap.put("2 dollar sign", "number2 en.dollar_sign");
		nlToEntityMap.put("3 dollar sign", "number3 en.dollar_sign");
		nlToEntityMap.put("4 dollar sign", "number4 en.dollar_sign");
		nlToEntityMap.put("5 dollar sign", "number5 en.dollar_sign");
		nlToEntityMap.put("1 dollar signs", "number1 en.dollar_sign");
		nlToEntityMap.put("2 dollar signs", "number2 en.dollar_sign");
		nlToEntityMap.put("3 dollar signs", "number3 en.dollar_sign");
		nlToEntityMap.put("4 dollar signs", "number4 en.dollar_sign");
		nlToEntityMap.put("5 dollar signs", "number5 en.dollar_sign");
		nlToEntityMap.put("1 dollar", "number1 en.dollar_sign");
		nlToEntityMap.put("2 dollar", "number2 en.dollar_sign");
		nlToEntityMap.put("3 dollar", "number3 en.dollar_sign");
		nlToEntityMap.put("4 dollar", "number4 en.dollar_sign");
		nlToEntityMap.put("5 dollar", "number5 en.dollar_sign");
		nlToEntityMap.put("1 menu", "number1 en.dollar_sign");
		nlToEntityMap.put("2 menu", "number2 en.dollar_sign");
		nlToEntityMap.put("3 menu", "number3 en.dollar_sign");
		nlToEntityMap.put("4 menu", "number4 en.dollar_sign");
		nlToEntityMap.put("5 menu", "number5 en.dollar_sign");
		nlToEntityMap.put("30 reviews", "number30 en.review");
		nlToEntityMap.put("40 reviews", "number40 en.review");
		*/
		
		
		return nlToEntityMap;
	}
	
	public static void main(String[] args) throws IOException {
		
		Map<String, String> nlMap = null;
		List<String> domains = Arrays.asList("recipes", "restaurants", "publications", "housing", "socialnetwork", "calendar","blocks", "basketball");

		for(String domain: domains) {
			if(domain.equals("recipes")){
				nlMap = getRecipesNLMap();
			}else if(domain.equals("restaurants")){
				nlMap = getRestaurantsNLMap();
			}else if(domain.equals("publications")){
				nlMap = getPublicationsNLMap();
			}else if(domain.equals("housing")){
				nlMap = getHousingNLMap();
			}else if(domain.equals("socialnetwork")){
				nlMap = getSocialNetworkNLMap();
			}else if(domain.equals("calendar")){
				nlMap = getCalendarNLMap();
			}else if(domain.equals("blocks")){
				nlMap = getBlocksNLMap();
			}else if(domain.equals("basketball")){
				nlMap = getBasketballNLMap();
			}
			
			if(nlMap == null){
				System.err.println(domain + " nl Map not initialized");
			}
			/*String sourceFolder;
			if(domain.equals("basketball") || domain.equals("socialnetwork")){
				 sourceFolder = "/home/parag/pCloudDrive/acl code/data_processing/prune4";
			}
			else {
			 sourceFolder = "/home/parag/pCloudDrive/acl code/data_processing/out1";
			}*/
			String sourceFolder = args[0];
			String destFolder = args[1];
			//String sourceFolder = "/home/parag/pCloudDrive/acl code/data_processing/out1";
			//String destFolder = "/home/parag/pCloudDrive/acl code/data_processing/java_out";


			 String sourceTrainFile = sourceFolder + File.separator + domain + "_train.tsv.prune.txt";
			 String sourceTestFile = sourceFolder + File.separator + domain + "_test.tsv.prune.txt";
			

			 String outputTrainFile = destFolder + File.separator + domain + "_train.tsv.entity.prune.txt";
			 String outputTestFile = destFolder + File.separator + domain + "_test.tsv.entity.prune.txt";
			
			 String outputTrainTransformationFile = destFolder + File.separator + domain + "_train.trans.txt";
			 String outputTestTransformationFile = destFolder + File.separator + domain + "_test.trans.txt";
			//transform train files.
			transform(domain, sourceTrainFile, outputTrainFile, outputTrainTransformationFile, nlMap);
			
			//transform test files.
			transform(domain,sourceTestFile, outputTestFile, outputTestTransformationFile, nlMap);
		}
		
	}

}
