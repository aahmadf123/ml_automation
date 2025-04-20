// Import necessary modules
const WebSocket = require('ws');
const { describe, it, before, after } = require('mocha');
const { expect } = require('chai');
const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');

// Set up Next.js
const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

// Start the server
let server;
const PORT = 3001; // Use a different port than the main app
const WS_PORT = 8000; // WebSocket port

before(async () => {
  await app.prepare();
  server = createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    handle(req, res, parsedUrl);
  });
  server.listen(PORT);
});

after(() => {
  server.close();
});

// Helper function to create a WebSocket connection
function createWebSocketConnection() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://localhost:${WS_PORT}`);
    
    ws.on('open', () => {
      resolve(ws);
    });
    
    ws.on('error', (error) => {
      reject(error);
    });
  });
}

// Helper function to wait for a message
function waitForMessage(ws, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('Timeout waiting for message'));
    }, timeout);
    
    ws.once('message', (data) => {
      clearTimeout(timer);
      resolve(JSON.parse(data));
    });
  });
}

describe('WebSocket Functionality', () => {
  describe('Connection', () => {
    it('should connect to the WebSocket server', async () => {
      const ws = await createWebSocketConnection();
      expect(ws.readyState).to.equal(WebSocket.OPEN);
      ws.close();
    });
    
    it('should receive a connection confirmation message', async () => {
      const ws = await createWebSocketConnection();
      
      const message = await waitForMessage(ws);
      expect(message).to.have.property('type');
      expect(message.type).to.equal('connection');
      expect(message).to.have.property('status');
      expect(message.status).to.equal('connected');
      
      ws.close();
    });
  });
  
  describe('System Metrics', () => {
    it('should receive system metrics updates', async () => {
      const ws = await createWebSocketConnection();
      
      // Wait for connection confirmation
      await waitForMessage(ws);
      
      // Wait for system metrics update
      const metricsMessage = await waitForMessage(ws);
      expect(metricsMessage).to.have.property('type');
      expect(metricsMessage.type).to.equal('system_metrics');
      expect(metricsMessage).to.have.property('data');
      
      const metrics = metricsMessage.data;
      expect(metrics).to.have.property('cpu');
      expect(metrics).to.have.property('memory');
      expect(metrics).to.have.property('disk');
      expect(metrics).to.have.property('network');
      
      // Check that the metrics have the expected structure
      expect(metrics.cpu).to.have.property('usage');
      expect(metrics.memory).to.have.property('used');
      expect(metrics.memory).to.have.property('total');
      expect(metrics.disk).to.have.property('used');
      expect(metrics.disk).to.have.property('total');
      expect(metrics.network).to.have.property('bytesIn');
      expect(metrics.network).to.have.property('bytesOut');
      
      ws.close();
    });
  });
  
  describe('Model Metrics', () => {
    it('should receive model metrics updates', async () => {
      const ws = await createWebSocketConnection();
      
      // Wait for connection confirmation
      await waitForMessage(ws);
      
      // Wait for model metrics update
      const metricsMessage = await waitForMessage(ws);
      expect(metricsMessage).to.have.property('type');
      expect(metricsMessage.type).to.equal('model_metrics');
      expect(metricsMessage).to.have.property('data');
      
      const metrics = metricsMessage.data;
      expect(metrics).to.have.property('modelId');
      expect(metrics).to.have.property('modelName');
      expect(metrics).to.have.property('rmse');
      expect(metrics).to.have.property('mse');
      expect(metrics).to.have.property('mae');
      expect(metrics).to.have.property('r2');
      expect(metrics).to.have.property('timestamp');
      
      ws.close();
    });
  });
  
  describe('Data Drift Alerts', () => {
    it('should receive data drift alerts', async () => {
      const ws = await createWebSocketConnection();
      
      // Wait for connection confirmation
      await waitForMessage(ws);
      
      // Wait for data drift alert
      const alertMessage = await waitForMessage(ws);
      expect(alertMessage).to.have.property('type');
      expect(alertMessage.type).to.equal('data_drift_alert');
      expect(alertMessage).to.have.property('data');
      
      const alert = alertMessage.data;
      expect(alert).to.have.property('id');
      expect(alert).to.have.property('feature');
      expect(alert).to.have.property('driftScore');
      expect(alert).to.have.property('threshold');
      expect(alert).to.have.property('timestamp');
      expect(alert).to.have.property('status');
      
      ws.close();
    });
  });
  
  describe('Pipeline Status', () => {
    it('should receive pipeline status updates', async () => {
      const ws = await createWebSocketConnection();
      
      // Wait for connection confirmation
      await waitForMessage(ws);
      
      // Wait for pipeline status update
      const statusMessage = await waitForMessage(ws);
      expect(statusMessage).to.have.property('type');
      expect(statusMessage.type).to.equal('pipeline_status');
      expect(statusMessage).to.have.property('data');
      
      const status = statusMessage.data;
      expect(status).to.have.property('stage');
      expect(status).to.have.property('status');
      expect(status).to.have.property('progress');
      expect(status).to.have.property('timestamp');
      
      ws.close();
    });
  });
  
  describe('Client Messages', () => {
    it('should send a message to the server and receive a response', async () => {
      const ws = await createWebSocketConnection();
      
      // Wait for connection confirmation
      await waitForMessage(ws);
      
      // Send a message to the server
      const clientMessage = {
        type: 'request',
        action: 'get_model_metrics',
        modelId: 'test-model-id'
      };
      
      ws.send(JSON.stringify(clientMessage));
      
      // Wait for response
      const responseMessage = await waitForMessage(ws);
      expect(responseMessage).to.have.property('type');
      expect(responseMessage.type).to.equal('response');
      expect(responseMessage).to.have.property('requestType');
      expect(responseMessage.requestType).to.equal('get_model_metrics');
      expect(responseMessage).to.have.property('data');
      
      const responseData = responseMessage.data;
      expect(responseData).to.have.property('modelId');
      expect(responseData.modelId).to.equal('test-model-id');
      expect(responseData).to.have.property('metrics');
      
      ws.close();
    });
  });
  
  describe('Multiple Clients', () => {
    it('should handle multiple client connections', async () => {
      // Create multiple WebSocket connections
      const ws1 = await createWebSocketConnection();
      const ws2 = await createWebSocketConnection();
      
      // Wait for connection confirmations
      await waitForMessage(ws1);
      await waitForMessage(ws2);
      
      // Send a message from the first client
      const clientMessage = {
        type: 'request',
        action: 'get_model_metrics',
        modelId: 'test-model-id'
      };
      
      ws1.send(JSON.stringify(clientMessage));
      
      // Wait for response on the first client
      const responseMessage1 = await waitForMessage(ws1);
      expect(responseMessage1).to.have.property('type');
      expect(responseMessage1.type).to.equal('response');
      
      // The second client should not receive the response
      // Set a short timeout to check that no message is received
      try {
        await waitForMessage(ws2, 1000);
        // If we get here, a message was received when it shouldn't have been
        expect.fail('Second client received a message when it shouldn\'t have');
      } catch (error) {
        // This is expected - timeout waiting for message
        expect(error.message).to.equal('Timeout waiting for message');
      }
      
      ws1.close();
      ws2.close();
    });
  });
  
  describe('Reconnection', () => {
    it('should handle client reconnection', async () => {
      // Create a WebSocket connection
      const ws = await createWebSocketConnection();
      
      // Wait for connection confirmation
      await waitForMessage(ws);
      
      // Close the connection
      ws.close();
      
      // Create a new connection
      const newWs = await createWebSocketConnection();
      
      // Wait for connection confirmation
      const message = await waitForMessage(newWs);
      expect(message).to.have.property('type');
      expect(message.type).to.equal('connection');
      expect(message).to.have.property('status');
      expect(message.status).to.equal('connected');
      
      newWs.close();
    });
  });
}); 