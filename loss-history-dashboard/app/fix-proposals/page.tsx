'use client';

import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Card, CardContent, Button, 
  TextField, Chip, Tabs, Tab, Dialog, 
  DialogTitle, DialogContent, DialogActions 
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CancelIcon from '@mui/icons-material/Cancel';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

export default function FixProposalsPage() {
  const [proposals, setProposals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState('pending');
  const [selectedProposal, setSelectedProposal] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [comment, setComment] = useState('');
  const [dialogAction, setDialogAction] = useState('');
  
  useEffect(() => {
    fetchProposals(selectedTab);
  }, [selectedTab]);
  
  const fetchProposals = async (status) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/fix-proposals?status=${status}`);
      const data = await response.json();
      if (data.success) {
        setProposals(data.data.map((p, index) => ({ id: index, ...p })));
      } else {
        console.error('Failed to fetch proposals:', data.error);
      }
    } catch (error) {
      console.error('Error fetching proposals:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };
  
  const openDialog = (proposal, action) => {
    setSelectedProposal(proposal);
    setDialogAction(action);
    setComment('');
    setDialogOpen(true);
  };
  
  const handleSubmitDecision = async () => {
    try {
      const response = await fetch('/api/fix-proposals', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          proposalId: selectedProposal.proposalId,
          decision: dialogAction,
          decidedBy: 'current-user', // Replace with actual user ID from auth
          comment: comment,
        }),
      });
      
      const data = await response.json();
      if (data.success) {
        // Refresh the list
        fetchProposals(selectedTab);
        setDialogOpen(false);
      } else {
        console.error('Failed to submit decision:', data.error);
      }
    } catch (error) {
      console.error('Error submitting decision:', error);
    }
  };
  
  const columns: GridColDef[] = [
    { field: 'proposalId', headerName: 'ID', width: 100 },
    { field: 'createdAt', headerName: 'Created', width: 180, 
      valueFormatter: (params) => new Date(params.value).toLocaleString() },
    { field: 'modelId', headerName: 'Model', width: 180 },
    { field: 'problemType', headerName: 'Type', width: 150 },
    { field: 'problem', headerName: 'Problem', width: 250 },
    { field: 'solution', headerName: 'Proposed Solution', width: 350 },
    { field: 'status', headerName: 'Status', width: 120,
      renderCell: (params) => {
        const value = params.value;
        if (value === 'approved') {
          return <Chip icon={<CheckCircleOutlineIcon />} label="Approved" color="success" />;
        } else if (value === 'rejected') {
          return <Chip icon={<CancelIcon />} label="Rejected" color="error" />;
        }
        return <Chip icon={<AccessTimeIcon />} label="Pending" color="warning" />;
      }
    },
    { field: 'actions', headerName: 'Actions', width: 200,
      renderCell: (params) => {
        if (params.row.status === 'pending') {
          return (
            <Box>
              <Button 
                variant="contained" 
                color="success" 
                size="small" 
                onClick={() => openDialog(params.row, 'approve')}
                sx={{ mr: 1 }}
              >
                Approve
              </Button>
              <Button 
                variant="contained" 
                color="error" 
                size="small"
                onClick={() => openDialog(params.row, 'reject')}
              >
                Reject
              </Button>
            </Box>
          );
        }
        return null;
      }
    },
  ];
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Fix Proposals
      </Typography>
      
      <Tabs value={selectedTab} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Pending" value="pending" />
        <Tab label="Approved" value="approved" />
        <Tab label="Rejected" value="rejected" />
        <Tab label="All" value="all" />
      </Tabs>
      
      <Card>
        <CardContent>
          <DataGrid
            rows={proposals}
            columns={columns}
            pageSize={10}
            rowsPerPageOptions={[5, 10, 25]}
            disableSelectionOnClick
            autoHeight
            loading={loading}
          />
        </CardContent>
      </Card>
      
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)}>
        <DialogTitle>
          {dialogAction === 'approve' ? 'Approve' : 'Reject'} Fix Proposal
        </DialogTitle>
        <DialogContent>
          {selectedProposal && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                Problem: {selectedProposal.problem}
              </Typography>
              <Typography variant="subtitle1" sx={{ mt: 2 }}>
                Proposed Solution: {selectedProposal.solution}
              </Typography>
              <TextField
                label="Comments (optional)"
                multiline
                rows={4}
                fullWidth
                variant="outlined"
                margin="normal"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)} color="inherit">
            Cancel
          </Button>
          <Button 
            onClick={handleSubmitDecision} 
            color={dialogAction === 'approve' ? 'success' : 'error'}
            variant="contained"
          >
            {dialogAction === 'approve' ? 'Approve' : 'Reject'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
} 