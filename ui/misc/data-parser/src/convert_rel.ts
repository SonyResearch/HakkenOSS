import csvParser from 'csv-parser';
import * as fs from 'fs';

interface Relation {
  ocid: string;
  name: string;
}
// Define the interface for the structure of the JSON elements
interface ConditionedRelation extends Relation {
  subject_domain: string;
  object_domain: string;
}
// Function to convert CSV to JSON format
function convertCsvToJson(csvFilePath: string, jsonFilePath: string): void {
  const result: Record<string, ConditionedRelation> = {};
  let relationCount = 1; // To create keys like relation1, relation2, ...

  // Read the CSV file with tab delimiter (\t)
  fs.createReadStream(csvFilePath)
    .pipe(csvParser({ separator: '\t' })) // Set the delimiter to \t (tab)
    .on('data', (row) => {
      const relationKey = `relation${relationCount}`;
      const relationName = (row.relation_type as string).toLowerCase().replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
      result[relationKey] = {
        name: relationName,
        ocid: `id${row.ocid_relation}`,
        subject_domain: (row.subject_domain as string).toLowerCase(),
        object_domain: (row.object_domain as string).toLowerCase(), 
      };
      relationCount++; // Increment for the next key
    })
    .on('end', () => {
      // Write the result to a JSON file
      fs.writeFileSync(jsonFilePath, JSON.stringify(result, null, 2));
      console.log('CSV successfully converted to JSON!');
    });
}

// Call the function, provide the path to the CSV file and output JSON file
const csvFilePath = '/path/to/your/edges_ui.csv'; // Change this to your CSV file path
const jsonFilePath =
  '/path/to/your/output/relations.ts'; // Change this to your desired JSON output file path

convertCsvToJson(csvFilePath, jsonFilePath);
