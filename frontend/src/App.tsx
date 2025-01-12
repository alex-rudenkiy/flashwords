// @ts-nocheck
import React, {useState} from 'react';
import {BrowserRouter as Router, Routes, Route, NavLink, Link} from 'react-router-dom';
import {QueryClient, QueryClientProvider} from 'react-query';
import * as Setting from "./Setting";

import {
    Container,
    Heading,
    Flex,
    Spacer,
    Box,
    Button,
    useDisclosure,
    IconButton,
} from '@chakra-ui/react';
import {Collapse} from '@chakra-ui/transition'; // Correct import here
import axios from 'axios';
import Review from './components/Review';
import {ChakraProvider} from '@chakra-ui/react';
import {createSystem, defaultConfig} from '@chakra-ui/react';
import {VocabularyList} from './components/VocabularyList';
import {IoClose} from "react-icons/io5";
import {RxHamburgerMenu} from "react-icons/rx";
import {AuthCallback} from "./components/AuthCallback";
import WordProgressChart from "./components/WordProgressChart";

export const system = createSystem(defaultConfig, {
    theme: {
        tokens: {
            fonts: {
                heading: {value: `'Figtree', sans-serif`},
                body: {value: `'Figtree', sans-serif`},
            },
        },
    },
});

const queryClient = new QueryClient();

axios.defaults.baseURL = '/api/';


function App() {

    // const { isOpen, onToggle, open, onOpen, onClose } = useDisclosure();
    const [isOpen, setIsOpen] = useState(false);

    return (
        <QueryClientProvider client={queryClient}>
            <ChakraProvider value={system}>
                <Router>
                    <Box as="nav">
                        {/*<Box as="nav" bg="gray.100" p={4} boxShadow="md">*/}

                        <Container maxW="container.md" display="flex" alignItems="center">
                            <Heading m={5} size="lg" as={NavLink} to="/"
                                     style={{textDecoration: 'none', color: 'black'}}>
                                Vocabulary App
                            </Heading>
                            <Spacer/>
                            <Flex display={{base: 'none', md: 'flex'}} gap={4} alignItems="center">
                                <Button as={NavLink} to="/" colorScheme="teal" variant="ghost">
                                    Vocabulary
                                </Button>
                                <Button as={NavLink} to="/review" colorScheme="teal" variant="ghost">
                                    Review
                                </Button>
                                <Button as={NavLink} to="/stats" colorScheme="teal" variant="ghost">
                                    Stats
                                </Button>
                                {!Setting.isLoggedIn() ?
                                    <Button colorScheme="teal" variant="ghost"
                                            onClick={() => Setting.CasdoorSDK.signin_redirect()}>
                                        Login
                                    </Button> : <Button colorScheme="teal" variant="ghost"
                                                        onClick={() => Setting.logout()}>
                                        Logout
                                    </Button>
                                }
                            </Flex>
                            <Box display={{base: 'block', md: 'none'}}>
                                <IconButton
                                    aria-label="Toggle Navigation"
                                    variant="ghost"
                                >
                                    {isOpen ?
                                        <IoClose onClick={() => setIsOpen(false)}/> :
                                        <RxHamburgerMenu onClick={() => setIsOpen(true)}/>}
                                </IconButton>
                            </Box>
                        </Container>
                        <Collapse in={isOpen} animateOpacity>
                            <Box p={4} bg="gray.100" display={{base: 'block', md: 'none'}}>
                                <Button as={NavLink} to="/" colorScheme="teal" variant="ghost" w="100%" my={2}>
                                    Vocabulary
                                </Button>
                                <Button as={NavLink} to="/review" colorScheme="teal" variant="ghost" w="100%" my={2}>
                                    Review
                                </Button>
                                <Button as={Link} to="/stats" colorScheme="teal" variant="ghost" w="100%" my={2}>
                                    Stats
                                </Button>
                                {!Setting.isLoggedIn() ?
                                    <Button colorScheme="teal" variant="ghost" w="100%" my={2}
                                            onClick={() => Setting.CasdoorSDK.signin_redirect()}>
                                        Login
                                    </Button> : <Button colorScheme="teal" variant="ghost" w="100%" my={2}
                                                        onClick={() => Setting.logout()}>
                                        Logout
                                    </Button>
                                }
                            </Box>
                        </Collapse>
                    </Box>
                    <Container maxW="container.md">
                        <Routes>
                            <Route path="/" element={<VocabularyList/>}/>
                            <Route path="/review" element={<Review/>}/>
                            {/*<Route path="/stats" element={<MetricsDashboard userId={1}/>}/>*/}
                            <Route path="/stats"
                                   element={<WordProgressChart userId={'5ef5f92f-6197-4ba7-8a7c-5dbd8f142828'}/>}/>
                            <Route path="/callback" element={<AuthCallback/>}/>
                        </Routes>
                    </Container>
                </Router>
            </ChakraProvider>
        </QueryClientProvider>
    );
}

export default App;
