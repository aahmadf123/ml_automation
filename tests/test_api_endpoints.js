// Import necessary modules
const fetch = require('node-fetch');
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

// Helper function to make API requests
async function apiRequest(endpoint, options = {}) {
  const url = `http://localhost:${PORT}/api${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  return response;
}

describe('API Endpoints', () => {
  describe('GET /api/metrics', () => {
    it('should return system metrics', async () => {
      const response = await apiRequest('/metrics');
      expect(response.status).to.equal(200);
      
      const data = await response.json();
      expect(data).to.be.an('object');
      expect(data).to.have.property('cpu');
      expect(data).to.have.property('memory');
      expect(data).to.have.property('disk');
      expect(data).to.have.property('network');
      
      // Check that the metrics have the expected structure
      expect(data.cpu).to.have.property('usage');
      expect(data.memory).to.have.property('used');
      expect(data.memory).to.have.property('total');
      expect(data.disk).to.have.property('used');
      expect(data.disk).to.have.property('total');
      expect(data.network).to.have.property('bytesIn');
      expect(data.network).to.have.property('bytesOut');
    });
  });
  
  describe('GET /api/models', () => {
    it('should return list of models', async () => {
      const response = await apiRequest('/models');
      expect(response.status).to.equal(200);
      
      const data = await response.json();
      expect(data).to.be.an('array');
      
      // Check that each model has the expected properties
      data.forEach(model => {
        expect(model).to.have.property('id');
        expect(model).to.have.property('name');
        expect(model).to.have.property('version');
        expect(model).to.have.property('status');
        expect(model).to.have.property('metrics');
      });
    });
  });
  
  describe('GET /api/models/:id/metrics', () => {
    it('should return metrics for a specific model', async () => {
      // First, get a list of models to find a valid ID
      const modelsResponse = await apiRequest('/models');
      const models = await modelsResponse.json();
      
      if (models.length > 0) {
        const modelId = models[0].id;
        const response = await apiRequest(`/models/${modelId}/metrics`);
        expect(response.status).to.equal(200);
        
        const data = await response.json();
        expect(data).to.be.an('object');
        expect(data).to.have.property('rmse');
        expect(data).to.have.property('mse');
        expect(data).to.have.property('mae');
        expect(data).to.have.property('r2');
        expect(data).to.have.property('history');
        
        // Check that history is an array
        expect(data.history).to.be.an('array');
        
        // Check that each history entry has the expected properties
        data.history.forEach(entry => {
          expect(entry).to.have.property('timestamp');
          expect(entry).to.have.property('rmse');
          expect(entry).to.have.property('mse');
          expect(entry).to.have.property('mae');
          expect(entry).to.have.property('r2');
        });
      }
    });
    
    it('should return 404 for non-existent model', async () => {
      const response = await apiRequest('/models/non-existent-id/metrics');
      expect(response.status).to.equal(404);
    });
  });
  
  describe('GET /api/alerts', () => {
    it('should return list of alerts', async () => {
      const response = await apiRequest('/alerts');
      expect(response.status).to.equal(200);
      
      const data = await response.json();
      expect(data).to.be.an('array');
      
      // Check that each alert has the expected properties
      data.forEach(alert => {
        expect(alert).to.have.property('id');
        expect(alert).to.have.property('title');
        expect(alert).to.have.property('message');
        expect(alert).to.have.property('type');
        expect(alert).to.have.property('priority');
        expect(alert).to.have.property('timestamp');
        expect(alert).to.have.property('status');
      });
    });
  });
  
  describe('POST /api/alerts', () => {
    it('should create a new alert', async () => {
      const newAlert = {
        title: 'Test Alert',
        message: 'This is a test alert',
        type: 'warning',
        priority: 'medium',
        source: 'test'
      };
      
      const response = await apiRequest('/alerts', {
        method: 'POST',
        body: JSON.stringify(newAlert)
      });
      
      expect(response.status).to.equal(201);
      
      const data = await response.json();
      expect(data).to.have.property('id');
      expect(data.title).to.equal(newAlert.title);
      expect(data.message).to.equal(newAlert.message);
      expect(data.type).to.equal(newAlert.type);
      expect(data.priority).to.equal(newAlert.priority);
      expect(data.source).to.equal(newAlert.source);
      expect(data).to.have.property('timestamp');
      expect(data).to.have.property('status');
    });
    
    it('should return 400 for invalid alert data', async () => {
      const invalidAlert = {
        // Missing required fields
        type: 'warning'
      };
      
      const response = await apiRequest('/alerts', {
        method: 'POST',
        body: JSON.stringify(invalidAlert)
      });
      
      expect(response.status).to.equal(400);
    });
  });
  
  describe('GET /api/dags', () => {
    it('should return list of DAGs', async () => {
      const response = await apiRequest('/dags');
      expect(response.status).to.equal(200);
      
      const data = await response.json();
      expect(data).to.be.an('array');
      
      // Check that each DAG has the expected properties
      data.forEach(dag => {
        expect(dag).to.have.property('id');
        expect(dag).to.have.property('name');
        expect(dag).to.have.property('status');
        expect(dag).to.have.property('lastRun');
        expect(dag).to.have.property('nextRun');
        expect(dag).to.have.property('schedule');
      });
    });
  });
  
  describe('GET /api/dags/:id/runs', () => {
    it('should return runs for a specific DAG', async () => {
      // First, get a list of DAGs to find a valid ID
      const dagsResponse = await apiRequest('/dags');
      const dags = await dagsResponse.json();
      
      if (dags.length > 0) {
        const dagId = dags[0].id;
        const response = await apiRequest(`/dags/${dagId}/runs`);
        expect(response.status).to.equal(200);
        
        const data = await response.json();
        expect(data).to.be.an('array');
        
        // Check that each run has the expected properties
        data.forEach(run => {
          expect(run).to.have.property('id');
          expect(run).to.have.property('dagId');
          expect(run).to.have.property('status');
          expect(run).to.have.property('startTime');
          expect(run).to.have.property('endTime');
          expect(run).to.have.property('duration');
        });
      }
    });
    
    it('should return 404 for non-existent DAG', async () => {
      const response = await apiRequest('/dags/non-existent-id/runs');
      expect(response.status).to.equal(404);
    });
  });
  
  describe('GET /api/mlflow/runs', () => {
    it('should return MLflow runs', async () => {
      const response = await apiRequest('/mlflow/runs');
      expect(response.status).to.equal(200);
      
      const data = await response.json();
      expect(data).to.be.an('array');
      
      // Check that each run has the expected properties
      data.forEach(run => {
        expect(run).to.have.property('id');
        expect(run).to.have.property('name');
        expect(run).to.have.property('status');
        expect(run).to.have.property('startTime');
        expect(run).to.have.property('endTime');
        expect(run).to.have.property('metrics');
        expect(run).to.have.property('parameters');
      });
    });
  });
  
  describe('GET /api/mlflow/runs/:id/metrics', () => {
    it('should return metrics for a specific MLflow run', async () => {
      // First, get a list of MLflow runs to find a valid ID
      const runsResponse = await apiRequest('/mlflow/runs');
      const runs = await runsResponse.json();
      
      if (runs.length > 0) {
        const runId = runs[0].id;
        const response = await apiRequest(`/mlflow/runs/${runId}/metrics`);
        expect(response.status).to.equal(200);
        
        const data = await response.json();
        expect(data).to.be.an('object');
        expect(data).to.have.property('rmse');
        expect(data).to.have.property('mse');
        expect(data).to.have.property('mae');
        expect(data).to.have.property('r2');
        expect(data).to.have.property('history');
        
        // Check that history is an array
        expect(data.history).to.be.an('array');
        
        // Check that each history entry has the expected properties
        data.history.forEach(entry => {
          expect(entry).to.have.property('timestamp');
          expect(entry).to.have.property('rmse');
          expect(entry).to.have.property('mse');
          expect(entry).to.have.property('mae');
          expect(entry).to.have.property('r2');
        });
      }
    });
    
    it('should return 404 for non-existent MLflow run', async () => {
      const response = await apiRequest('/mlflow/runs/non-existent-id/metrics');
      expect(response.status).to.equal(404);
    });
  });
}); 