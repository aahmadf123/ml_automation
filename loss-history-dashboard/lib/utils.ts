import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Convert a readable stream to a string
 * Used for S3 response bodies which are returned as streams
 * 
 * @param stream The readable stream to convert
 * @param timeout Optional timeout in milliseconds (defaults to 30000)
 * @returns Promise that resolves to the string content
 */
export async function streamToString(stream: any, timeout: number = 30000): Promise<string> {
  if (!stream) {
    console.warn('Stream was undefined or null in streamToString')
    return '';
  }
  
  return new Promise((resolve, reject) => {
    const chunks: Uint8Array[] = [];
    let timeoutId: NodeJS.Timeout | null = null;
    
    // Set timeout if specified
    if (timeout > 0) {
      timeoutId = setTimeout(() => {
        cleanup();
        reject(new Error(`Stream conversion timed out after ${timeout}ms`));
      }, timeout);
    }
    
    // Handle stream events
    stream.on('data', (chunk: Uint8Array) => chunks.push(chunk));
    
    stream.on('error', (err: Error) => {
      cleanup();
      console.error('Error in stream conversion:', err);
      reject(err);
    });
    
    stream.on('end', () => {
      cleanup();
      try {
        const result = Buffer.concat(chunks).toString('utf-8');
        resolve(result);
      } catch (err) {
        console.error('Error converting buffer to string:', err);
        reject(err);
      }
    });
    
    // Cleanup function to remove listeners and clear timeout
    function cleanup() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      
      // Remove listeners to prevent memory leaks
      stream.removeAllListeners('data');
      stream.removeAllListeners('error');
      stream.removeAllListeners('end');
    }
  });
}

/**
 * Format a number as currency
 * 
 * @param value Number to format
 * @param currency Currency code (default: 'USD')
 * @param maximumFractionDigits Maximum fraction digits (default: 0)
 * @returns Formatted currency string
 */
export function formatCurrency(value: number, currency: string = 'USD', maximumFractionDigits: number = 0): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits
  }).format(value);
}

/**
 * Format a number in millions (e.g., "$5M")
 * 
 * @param value Number to format
 * @returns Formatted string in millions
 */
export function formatMillions(value: number): string {
  return `$${(value / 1000000).toFixed(1)}M`;
}
