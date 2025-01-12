//@ts-nocheck
import React from 'react';
import { Box, Typography } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

const DataTable = ({ topWords, predictions }: {Array, Array}) => {
  const columns = [
    { field: 'word', headerName: 'Word', flex: 1 },
    { field: 'correct', headerName: 'Correct Answers', flex: 1 },
    { field: 'incorrect', headerName: 'Incorrect Answers', flex: 1 },
    { field: 'learned', headerName: 'Learned Level', flex: 1 },
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Top Words</Typography>
      <DataGrid
        rows={topWords}
        columns={columns}
        autoHeight
        pageSize={5}
        rowsPerPageOptions={[5, 10]}
      />

      <Typography variant="h6" gutterBottom>Predictions</Typography>
      <DataGrid
        rows={predictions}
        columns={[{ field: 'word', headerName: 'Word', flex: 1 }, { field: 'score', headerName: 'Score', flex: 1 }]}
        autoHeight
        pageSize={5}
        rowsPerPageOptions={[5, 10]}
      />
    </Box>
  );
};

export default DataTable;
