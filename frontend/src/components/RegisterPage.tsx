//@ts-nocheck
import React, {useState, useEffect} from "react";
import { Box, Input, Button, VStack, Text } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { register } from '../api/auth';

function RegisterPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleRegister = async () => {
    try {
      await register({ username, password });
      navigate('/login');
    } catch (err) {
      setError('Registration failed.');
    }
  };

  return (
    <VStack spacing={4} p={6} boxShadow="lg" borderRadius="md" w={{ base: '90%', md: '400px' }}>
      <Text fontSize="xl">Register</Text>
      {error && <Text color="red.500">{error}</Text>}
      <Input
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />
      <Input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <Button colorScheme="blue" onClick={handleRegister}>Register</Button>
    </VStack>
  );
}

export default RegisterPage;