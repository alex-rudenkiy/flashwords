//@ts-nocheck
import React from 'react';
import { Box, Typography } from '@mui/material';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const Charts = ({ data, frequency }: {any, any}) => {
  const { correct, incorrect, learned, unlearned } = data;

  const pieData = [
    { name: 'Correct', value: correct },
    { name: 'Incorrect', value: incorrect },
    { name: 'Learned', value: learned },
    { name: 'Unlearned', value: unlearned },
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Word Statistics</Typography>
      <PieChart width={400} height={300}>
        <Pie
          data={pieData}
          cx={200}
          cy={150}
          outerRadius={80}
          fill="#8884d8"
          dataKey="value"
          label
        >
          {pieData?.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>

      <Typography variant="h6" gutterBottom>Frequency of Words</Typography>
      <BarChart width={400} height={300} data={frequency}>
        <XAxis dataKey="period" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="frequency" fill="#82ca9d" />
      </BarChart>
    </Box>
  );
};

export default Charts;
