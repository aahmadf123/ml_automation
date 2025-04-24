import { NextResponse } from 'next/server';
import { S3Client, GetObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { streamToString } from '@/lib/utils';

// Configure AWS SDK
const s3Client = new S3Client({
  region: process.env.AWS_REGION || 'us-east-1',
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
  },
});

// Define the S3 bucket where projections are stored
const BUCKET_NAME = process.env.DATA_BUCKET || 'grange-seniordesign-bucket';
// This is the path to the latest projections JSON file in S3
const LATEST_PROJECTIONS_KEY = 'projections/latest_projections.json';
// This is the prefix for the projections folder in S3
const PROJECTIONS_PREFIX = 'projections/';

export async function GET(request: Request) {
  try {
    // Extract query parameters for filtering
    const { searchParams } = new URL(request.url);
    const region = searchParams.get('region');
    const product = searchParams.get('product');
    const history = searchParams.get('history');
    const limit = parseInt(searchParams.get('limit') || '5', 10);
    
    // If history=true, return a list of available projection files
    if (history === 'true') {
      return await getProjectionHistory(limit);
    }
    
    console.log(`Fetching projections from S3: s3://${BUCKET_NAME}/${LATEST_PROJECTIONS_KEY}`);
    console.log(`Filters - Region: ${region}, Product: ${product}`);
    
    // Create a GetObject command
    const command = new GetObjectCommand({
      Bucket: BUCKET_NAME,
      Key: LATEST_PROJECTIONS_KEY,
    });
    
    try {
      // Get the object from S3
      const response = await s3Client.send(command);
      
      // Convert the stream to a string
      const jsonString = await streamToString(response.Body);
      
      // Parse the JSON data
      const projectionsData = JSON.parse(jsonString);
      
      // Validate the data structure
      if (!projectionsData || typeof projectionsData !== 'object') {
        return NextResponse.json(
          { 
            error: 'Invalid data format', 
            message: 'The projections data does not have the expected format.' 
          }, 
          { status: 400 }
        );
      }
      
      // Check if region exists in data
      if (region && (!projectionsData.historical_data || !projectionsData.historical_data[region])) {
        return NextResponse.json(
          { 
            error: 'Invalid region', 
            message: `Region "${region}" not found in the projections data.`,
            availableRegions: Object.keys(projectionsData.historical_data || {})
          }, 
          { status: 400 }
        );
      }
      
      // Check if product exists for the selected region
      if (region && product && (!projectionsData.historical_data[region] || !projectionsData.historical_data[region][product])) {
        return NextResponse.json(
          { 
            error: 'Invalid product', 
            message: `Product "${product}" not found for region "${region}" in the projections data.`,
            availableProducts: Object.keys(projectionsData.historical_data[region] || {})
          }, 
          { status: 400 }
        );
      }
      
      // Apply filters if specified
      let filteredData = { ...projectionsData };
      
      if (region) {
        // Filter historical data
        if (filteredData.historical_data) {
          filteredData.historical_data = {
            [region]: filteredData.historical_data[region] || {}
          };
        }
        
        // Filter projected data
        if (filteredData.projected_data) {
          filteredData.projected_data = {
            [region]: filteredData.projected_data[region] || {}
          };
        }
        
        // Filter decile data
        if (filteredData.decile_data) {
          filteredData.decile_data = {
            [region]: filteredData.decile_data[region] || {}
          };
        }
      }
      
      if (product && region) {
        // Further filter by product if region is specified
        if (filteredData.historical_data && filteredData.historical_data[region]) {
          filteredData.historical_data[region] = {
            [product]: filteredData.historical_data[region][product] || []
          };
        }
        
        if (filteredData.projected_data && filteredData.projected_data[region]) {
          filteredData.projected_data[region] = {
            [product]: filteredData.projected_data[region][product] || []
          };
        }
        
        if (filteredData.decile_data && filteredData.decile_data[region]) {
          filteredData.decile_data[region] = {
            [product]: filteredData.decile_data[region][product] || []
          };
        }
      }
      
      // Return the filtered projections data
      return NextResponse.json({
        success: true,
        data: filteredData
      }, { status: 200 });
      
    } catch (error: any) {
      console.error("Error fetching from S3:", error);
      
      // Check if it's a NoSuchKey error (file not found)
      if (error.name === 'NoSuchKey') {
        return NextResponse.json(
          { 
            error: 'Projections not found', 
            message: 'No projections data has been generated yet.',
            s3Path: `s3://${BUCKET_NAME}/${LATEST_PROJECTIONS_KEY}`
          }, 
          { status: 404 }
        );
      }
      
      // For authentication/authorization errors
      if (error.name === 'AccessDenied') {
        return NextResponse.json(
          { 
            error: 'Access denied', 
            message: 'Not authorized to access the projections. Check AWS credentials and IAM permissions for the S3 bucket.',
            requiredPermissions: ['s3:GetObject', 's3:ListBucket']
          }, 
          { status: 403 }
        );
      }
      
      // For other errors
      return NextResponse.json(
        { 
          error: 'Failed to fetch projections', 
          message: error.message || 'An unknown error occurred',
          details: error.stack ? error.stack.split('\n')[0] : null
        }, 
        { status: 500 }
      );
    }
  } catch (error: any) {
    console.error("API route error:", error);
    return NextResponse.json(
      { 
        error: 'Internal server error', 
        message: error.message || 'An unknown error occurred',
        details: error.stack ? error.stack.split('\n')[0] : null
      }, 
      { status: 500 }
    );
  }
}

/**
 * Get the history of projection files from S3
 * 
 * @param limit The maximum number of files to return
 * @returns Response with the list of projection files
 */
async function getProjectionHistory(limit: number = 5) {
  try {
    // List objects in the projections directory
    const command = new ListObjectsV2Command({
      Bucket: BUCKET_NAME,
      Prefix: PROJECTIONS_PREFIX,
      MaxKeys: 100 // Get more than we need to filter for actual projections files
    });
    
    const response = await s3Client.send(command);
    
    if (!response.Contents || response.Contents.length === 0) {
      return NextResponse.json(
        {
          success: true,
          data: []
        },
        { status: 200 }
      );
    }
    
    // Filter for model_projections files only and sort by last modified date (newest first)
    const projectionFiles = response.Contents
      .filter(item => 
        item.Key && 
        item.Key.includes('model_projections_') && 
        item.Key.endsWith('.json')
      )
      .sort((a, b) => {
        // Sort by LastModified in descending order (newest first)
        const dateA = a.LastModified ? new Date(a.LastModified).getTime() : 0;
        const dateB = b.LastModified ? new Date(b.LastModified).getTime() : 0;
        return dateB - dateA;
      })
      .slice(0, limit) // Get only the requested number of files
      .map(item => ({
        key: item.Key,
        size: item.Size,
        lastModified: item.LastModified,
        url: `s3://${BUCKET_NAME}/${item.Key}`,
        // Extract the timestamp from the file name
        timestamp: item.Key ? item.Key.match(/model_projections_(\d+_\d+)\.json/)?.[1] : null
      }));
    
    return NextResponse.json(
      {
        success: true,
        data: projectionFiles
      },
      { status: 200 }
    );
  } catch (error: any) {
    console.error("Error listing projection files:", error);
    
    // For authentication/authorization errors
    if (error.name === 'AccessDenied') {
      return NextResponse.json(
        { 
          error: 'Access denied', 
          message: 'Not authorized to list objects in the S3 bucket. Check AWS credentials and IAM permissions.',
          requiredPermissions: ['s3:ListBucket']
        }, 
        { status: 403 }
      );
    }
    
    return NextResponse.json(
      { 
        error: 'Failed to list projection files', 
        message: error.message || 'An unknown error occurred',
        details: error.stack ? error.stack.split('\n')[0] : null
      }, 
      { status: 500 }
    );
  }
} 