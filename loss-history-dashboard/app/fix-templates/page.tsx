"use client"

import React, { useState, useEffect } from 'react';
import {
  Box, Typography, Card, CardContent, Grid, Button,
  TextField, Dialog, DialogTitle, DialogContent,
  DialogActions, IconButton, Snackbar, Alert
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

export default function FixTemplatesPage() {
  const [templates, setTemplates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [currentTemplate, setCurrentTemplate] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    problemType: '',
    solution: '',
    code: ''
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/fix-templates');
      const data = await response.json();
      if (data.success) {
        setTemplates(data.data);
      } else {
        console.error('Failed to fetch templates:', data.error);
      }
    } catch (error) {
      console.error('Error fetching templates:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const openCreateDialog = () => {
    setIsEditing(false);
    setFormData({
      name: '',
      description: '',
      problemType: '',
      solution: '',
      code: ''
    });
    setDialogOpen(true);
  };

  const openEditDialog = (template) => {
    setIsEditing(true);
    setCurrentTemplate(template);
    setFormData({
      name: template.name,
      description: template.description,
      problemType: template.problemType,
      solution: template.solution,
      code: template.code || ''
    });
    setDialogOpen(true);
  };

  const handleSubmit = async () => {
    // Validate form
    if (!formData.name || !formData.problemType || !formData.solution) {
      setSnackbar({
        open: true,
        message: 'Please fill in all required fields',
        severity: 'error'
      });
      return;
    }

    try {
      const method = isEditing ? 'PUT' : 'POST';
      const url = isEditing 
        ? `/api/fix-templates/${currentTemplate.id}`
        : '/api/fix-templates';
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      const data = await response.json();
      if (data.success) {
        setDialogOpen(false);
        fetchTemplates();
        setSnackbar({
          open: true,
          message: `Template ${isEditing ? 'updated' : 'created'} successfully`,
          severity: 'success'
        });
      } else {
        setSnackbar({
          open: true,
          message: data.error || `Failed to ${isEditing ? 'update' : 'create'} template`,
          severity: 'error'
        });
      }
    } catch (error) {
      console.error('Error submitting form:', error);
      setSnackbar({
        open: true,
        message: `Error: ${error.message}`,
        severity: 'error'
      });
    }
  };

  const handleDelete = async (templateId) => {
    if (window.confirm('Are you sure you want to delete this template?')) {
      try {
        const response = await fetch(`/api/fix-templates/${templateId}`, {
          method: 'DELETE',
        });
        
        const data = await response.json();
        if (data.success) {
          fetchTemplates();
          setSnackbar({
            open: true,
            message: 'Template deleted successfully',
            severity: 'success'
          });
        } else {
          setSnackbar({
            open: true,
            message: data.error || 'Failed to delete template',
            severity: 'error'
          });
        }
      } catch (error) {
        console.error('Error deleting template:', error);
        setSnackbar({
          open: true,
          message: `Error: ${error.message}`,
          severity: 'error'
        });
      }
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar((prev) => ({
      ...prev,
      open: false
    }));
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Fix Templates
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<AddIcon />}
          onClick={openCreateDialog}
        >
          Add Template
        </Button>
      </Box>
      
      {loading ? (
        <Typography>Loading templates...</Typography>
      ) : (
        <Grid container spacing={3}>
          {templates.map((template) => (
            <Grid item xs={12} md={6} lg={4} key={template.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Typography variant="h6" gutterBottom>
                      {template.name}
                    </Typography>
                    <Box>
                      <IconButton size="small" onClick={() => openEditDialog(template)}>
                        <EditIcon />
                      </IconButton>
                      <IconButton size="small" onClick={() => handleDelete(template.id)}>
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Problem Type: {template.problemType}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    {template.description}
                  </Typography>
                  <Typography variant="subtitle2" sx={{ mt: 2 }}>
                    Solution:
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {template.solution}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      
      <Dialog 
        open={dialogOpen} 
        onClose={() => setDialogOpen(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>
          {isEditing ? 'Edit Template' : 'Create New Template'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              name="name"
              label="Template Name"
              fullWidth
              margin="normal"
              value={formData.name}
              onChange={handleInputChange}
              required
            />
            <TextField
              name="problemType"
              label="Problem Type"
              fullWidth
              margin="normal"
              value={formData.problemType}
              onChange={handleInputChange}
              required
              helperText="E.g., Data Drift, Feature Importance Change, etc."
            />
            <TextField
              name="description"
              label="Description"
              fullWidth
              multiline
              rows={2}
              margin="normal"
              value={formData.description}
              onChange={handleInputChange}
            />
            <TextField
              name="solution"
              label="Solution Template"
              fullWidth
              multiline
              rows={4}
              margin="normal"
              value={formData.solution}
              onChange={handleInputChange}
              required
              helperText="Description of the solution approach"
            />
            <TextField
              name="code"
              label="Code Template (optional)"
              fullWidth
              multiline
              rows={6}
              margin="normal"
              value={formData.code}
              onChange={handleInputChange}
              helperText="Any example code for implementation"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)} color="inherit">
            Cancel
          </Button>
          <Button 
            onClick={handleSubmit} 
            color="primary"
            variant="contained"
          >
            {isEditing ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
