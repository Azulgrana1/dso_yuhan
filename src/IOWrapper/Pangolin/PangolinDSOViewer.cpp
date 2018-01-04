/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"
#include <boost/algorithm/string.hpp>

#include "util/settings.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include <fstream>
#include <iostream>
#include <dirent.h>


namespace dso
{
namespace IOWrap
{



PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread)
{
	this->w = w;
	this->h = h;
	running=true;

	dataDir = "/home/wyw/lyh/data/slam";

	{
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		internalVideoImg = new MinimalImageB3(w,h);
		internalKFImg = new MinimalImageB3(w,h);
		internalResImg = new MinimalImageB3(w,h);
		videoImgChanged=kfImgChanged=resImgChanged=true;

		internalVideoImg->setBlack();
		internalKFImg->setBlack();
		internalResImg->setBlack();
	}


	{
		currentCam = new KeyFrameDisplay();
	}

	needReset = false;


    if(startRunThread)
        runThread = boost::thread(&PangolinDSOViewer::run, this);

}


PangolinDSOViewer::~PangolinDSOViewer()
{
	close();
	runThread.join();
}

void PangolinDSOViewer::addObjCamConnection(const char* filename)
{
	std::ifstream objCamFile;
	objCamFile.open(filename);

	std::vector<std::string> alllines;
	std::string sline;
	while( getline(objCamFile,sline) )
	{
		alllines.push_back(sline);
	}

	int nConnections = alllines.size();

	std::vector<std::string> strs;
	for(int i = 0;i<alllines.size();i++){


		boost::split(strs, alllines[i],boost::is_any_of(" "));
		std::cout<<strs[0]<<' '<<strs[1]<<std::endl;
		if(strs.size() < 2 || strs.size() > 3){
			printf("Error in read obj-cam file\n");
			exit(1);
		}


		objConnections.push_back(ObjCamConnection(atoi(strs[0].c_str()), atoi(strs[1].c_str()) ) );
	}

	objCamFile.close();
}

void PangolinDSOViewer::run()
{
	printf("START PANGOLIN!\n");

	pangolin::CreateWindowAndBind("Main",2*w,2*h);
	const int UI_WIDTH = 180;

	glEnable(GL_DEPTH_TEST);

	// 3D visualization
	pangolin::OpenGlRenderState Visualization3D_camera(
		pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000),
		pangolin::ModelViewLookAt(-0,-5,-10, 0,0,0, pangolin::AxisNegY)
		);

	pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h)
		.SetHandler(new pangolin::Handler3D(Visualization3D_camera));


	// 3 images
	pangolin::View& d_kfDepth = pangolin::Display("imgKFDepth")
	    .SetAspect(w/(float)h);

	pangolin::View& d_video = pangolin::Display("imgVideo")
	    .SetAspect(w/(float)h);

	pangolin::View& d_residual = pangolin::Display("imgResidual")
	    .SetAspect(w/(float)h);

	pangolin::GlTexture texKFDepth(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::GlTexture texVideo(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::GlTexture texResidual(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);


    pangolin::CreateDisplay()
		  .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
		  .SetLayout(pangolin::LayoutEqual)
		  .AddDisplay(d_kfDepth)
		  .AddDisplay(d_video)
		  .AddDisplay(d_residual);

	// parameter reconfigure gui
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

	pangolin::Var<int> settings_pointCloudMode("ui.PC_mode",1,1,4,false);

	pangolin::Var<bool> settings_showKFCameras("ui.KFCam",false,true);
	pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam",true,true);
	pangolin::Var<bool> settings_showTrajectory("ui.Trajectory",true,true);
	pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory",false,true);
	pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst",true,true);
	pangolin::Var<bool> settings_showAllConstraints("ui.AllConst",false,true);
	pangolin::Var<bool> settings_showObjectConstraints("ui.ObjConst", true, true);


	pangolin::Var<bool> settings_show3D("ui.show3D",true,true);
	pangolin::Var<bool> settings_showLiveDepth("ui.showDepth",false,true);
	pangolin::Var<bool> settings_showLiveVideo("ui.showVideo",false,true);
    pangolin::Var<bool> settings_showLiveResidual("ui.showResidual",false,true);

	pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow",false,true);
	pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking",false,true);
	pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking",false,true);


	pangolin::Var<int> settings_sparsity("ui.sparsity",18,1,20,false);
	pangolin::Var<int> settings_skipframes("ui.skipframes",0,0,100,false);
	pangolin::Var<double> settings_scaledVarTH("ui.relVarTH",0.001,1e-10,1e10, true);
	pangolin::Var<double> settings_absVarTH("ui.absVarTH",0.001,1e-10,1e10, true);
	pangolin::Var<double> settings_minRelBS("ui.minRelativeBS",0.1,0,1, false);


	pangolin::Var<bool> settings_resetButton("ui.Reset",false,false);
    pangolin::Var<bool> settings_saveButton("ui.SaveAll",false,false);

	pangolin::Var<int> settings_nPts("ui.activePoints",setting_desiredPointDensity, 50,5000, false);
	pangolin::Var<int> settings_nCandidates("ui.pointCandidates",setting_desiredImmatureDensity, 50,5000, false);
	pangolin::Var<int> settings_nMaxFrames("ui.maxFrames",setting_maxFrames, 4,10, false);
	pangolin::Var<double> settings_kfFrequency("ui.kfFrequency",setting_kfGlobalWeight,0.1,3, false);
	pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd",setting_minGradHistAdd,0,15, false);

	pangolin::Var<double> settings_trackFps("ui.Track fps",0,0,0,false);
	pangolin::Var<double> settings_mapFps("ui.KF fps",0,0,0,false);


	// Default hooks for exiting (Esc) and fullscreen (tab).
	while( !pangolin::ShouldQuit() && running )
	{
		// Clear entire screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if(setting_render_display3D)
		{
			// Activate efficiently by object
			Visualization3D_display.Activate(Visualization3D_camera);
			boost::unique_lock<boost::mutex> lk3d(model3DMutex);
			//pangolin::glDrawColouredCube();
			int refreshed=0;
			int frameCount = 0;
			for(KeyFrameDisplay* fh : keyframes)
			{
				frameCount++;
				if(frameCount<this->settings_skipframes)continue;							
				float blue[3] = {0,0,1};
				if(this->settings_showKFCameras) fh->drawCam(1,blue,0.1);


				refreshed =+ (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
						this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
				fh->drawPC(1);
			}
			for(KeyFrameDisplay* fh : objects)
			{
				fh->drawObj();
			}
			if(this->settings_showCurrentCamera) currentCam->drawCam(2,0,0.1);
			drawConstraints();
			lk3d.unlock();
		}



		openImagesMutex.lock();
		if(videoImgChanged) 	texVideo.Upload(internalVideoImg->data,GL_BGR,GL_UNSIGNED_BYTE);
		if(kfImgChanged) 		texKFDepth.Upload(internalKFImg->data,GL_BGR,GL_UNSIGNED_BYTE);
		if(resImgChanged) 		texResidual.Upload(internalResImg->data,GL_BGR,GL_UNSIGNED_BYTE);
		videoImgChanged=kfImgChanged=resImgChanged=false;
		openImagesMutex.unlock();




		// update fps counters
		{
			openImagesMutex.lock();
			float sd=0;
			for(float d : lastNMappingMs) sd+=d;
			settings_mapFps=lastNMappingMs.size()*1000.0f / sd;
			openImagesMutex.unlock();
		}
		{
			model3DMutex.lock();
			float sd=0;
			for(float d : lastNTrackingMs) sd+=d;
			settings_trackFps = lastNTrackingMs.size()*1000.0f / sd;
			model3DMutex.unlock();
		}


		if(setting_render_displayVideo)
		{
			d_video.Activate();
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			texVideo.RenderToViewportFlipY();
		}

		if(setting_render_displayDepth)
		{
			d_kfDepth.Activate();
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			texKFDepth.RenderToViewportFlipY();
		}

		if(setting_render_displayResidual)
		{
			d_residual.Activate();
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			texResidual.RenderToViewportFlipY();
		}


	    // update parameters
	    this->settings_pointCloudMode = settings_pointCloudMode.Get();

	    this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
	    this->settings_showAllConstraints = settings_showAllConstraints.Get();
	    this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
	    this->settings_showKFCameras = settings_showKFCameras.Get();
	    this->settings_showTrajectory = settings_showTrajectory.Get();
	    this->settings_showFullTrajectory = settings_showFullTrajectory.Get();
	    this->settings_showObjectConstraints = settings_showObjectConstraints.Get();

		setting_render_display3D = settings_show3D.Get();
		setting_render_displayDepth = settings_showLiveDepth.Get();
		setting_render_displayVideo =  settings_showLiveVideo.Get();
		setting_render_displayResidual = settings_showLiveResidual.Get();

		setting_render_renderWindowFrames = settings_showFramesWindow.Get();
		setting_render_plotTrackingFull = settings_showFullTracking.Get();
		setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();


	    this->settings_absVarTH = settings_absVarTH.Get();
	    this->settings_scaledVarTH = settings_scaledVarTH.Get();
	    this->settings_minRelBS = settings_minRelBS.Get();
	    this->settings_sparsity = settings_sparsity.Get();
	    this->settings_skipframes = settings_skipframes.Get();


	    setting_desiredPointDensity = settings_nPts.Get();
	    setting_desiredImmatureDensity = settings_nCandidates.Get();
	    setting_maxFrames = settings_nMaxFrames.Get();
	    setting_kfGlobalWeight = settings_kfFrequency.Get();
	    setting_minGradHistAdd = settings_gradHistAdd.Get();


	    if(settings_resetButton.Get())
	    {
	    	printf("RESET!\n");
	    	settings_resetButton.Reset();
	    	setting_fullResetRequested = true;
	    }
	    if(settings_saveButton.Get()){
	    	//save All
	    	printf("SAVE ALL!\n");
	    	settings_saveButton.Reset();
	    	saveAll();
	    }

		// Swap frames and Process Events
		pangolin::FinishFrame();


	    if(needReset) reset_internal();
	}


	printf("QUIT Pangolin thread!\n");
	printf("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");

	exit(1);
}


void PangolinDSOViewer::close()
{
	running = false;
}

void PangolinDSOViewer::join()
{
	runThread.join();
	printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
	needReset = true;
}

void PangolinDSOViewer::reset_internal()
{
	model3DMutex.lock();
	for(size_t i=0; i<keyframes.size();i++) delete keyframes[i];
	keyframes.clear();
	allFramePoses.clear();
	keyframesByKFID.clear();
	connections.clear();
	model3DMutex.unlock();


	openImagesMutex.lock();
	internalVideoImg->setBlack();
	internalKFImg->setBlack();
	internalResImg->setBlack();
	videoImgChanged= kfImgChanged= resImgChanged=true;
	openImagesMutex.unlock();

	needReset = false;
}


void PangolinDSOViewer::drawConstraints()
{
	if(settings_showAllConstraints)
	{
		// draw constraints
		glLineWidth(1);
		glBegin(GL_LINES);

		glColor3f(0,1,0);
		glBegin(GL_LINES);		
		for(unsigned int i=0;i<connections.size();i++)
		{
			if(connections[i].to == 0 || connections[i].from==0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;
			int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
			if(nAct==0 && nMarg>0)
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			}
		}
		glEnd();
	}

	if(settings_showActiveConstraints)
	{
		glLineWidth(3);
		glColor3f(0,0,1);
		glBegin(GL_LINES);
		for(unsigned int i=0;i<connections.size();i++)
		{
			if(connections[i].to == 0 || connections[i].from==0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;

			if(nAct>0)
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			}
		}
		glEnd();
	}

	if(settings_showTrajectory)
	{
		float colorRed[3] = {1,0,0};
		glColor3f(colorRed[0],colorRed[1],colorRed[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for(unsigned int i=0;i<keyframes.size();i++)
		{
			glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
					(float)keyframes[i]->camToWorld.translation()[1],
					(float)keyframes[i]->camToWorld.translation()[2]);
		}
		glEnd();
	}

	if(settings_showFullTrajectory)
	{
		float colorGreen[3] = {0,1,0};
		glColor3f(colorGreen[0],colorGreen[1],colorGreen[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for(unsigned int i=0;i<allFramePoses.size();i++)
		{
			glVertex3f((float)allFramePoses[i][0],
					(float)allFramePoses[i][1],
					(float)allFramePoses[i][2]);
		}
		glEnd();
	}

	if(settings_showObjectConstraints)
	{
		int objID;
		int frameID;
		bool objFound;
		bool frameFound;

		glColor3f(0, 1, 0);
		glLineWidth(3);

		glBegin(GL_LINES);
		for(unsigned int i = 0; i < objConnections.size(); i++)
		{
			objID = objConnections[i].objID + 20000;
			frameID = objConnections[i].frameID;
			objFound = objectsById.count(objID);
			frameFound = keyframesByKFID.count(frameID);

			if(!objFound || !frameFound)
				break;

			Sophus::Vector3f t = keyframesByKFID[frameID]->camToWorld.translation().cast<float>();
			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			t = objectsById[objID]->camToWorld.translation().cast<float>();
			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
		}
		glEnd();
	}
}




        
void PangolinDSOViewer::publishGraph(const std::map<uint64_t,Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Vector2i> > > &connectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;
    boost::unique_lock<boost::mutex> lk(model3DMutex);
	//model3DMutex.lock();
    connections.resize(connectivity.size());   

	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if(host > target) continue;


		connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connections[runningID].fwdAct = p.second[0];
		connections[runningID].fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		Eigen::Vector2i st = connectivity.at(inverseKey);
		connections[runningID].bwdAct = st[0];
		connections[runningID].bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}


	//model3DMutex.unlock();
}
void PangolinDSOViewer::publishKeyframes(
		std::vector<FrameHessian*> &frames,
		bool final,
		CalibHessian* HCalib)
{
	if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	for(FrameHessian* fh : frames)
	{
		if(keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
		{
			KeyFrameDisplay* kfd = new KeyFrameDisplay();
			keyframesByKFID[fh->frameID] = kfd;
			keyframes.push_back(kfd);
		}
		keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
	}
}
void PangolinDSOViewer::publishCamPose(FrameShell* frame,
		CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
	if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
	last_track = time_now;

	if(!setting_render_display3D) return;

	currentCam->setFromF(frame, HCalib);
	allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
}


void PangolinDSOViewer::pushLiveFrame(FrameHessian* image)
{
	if(!setting_render_displayVideo) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	for(int i=0;i<w*h;i++)
		internalVideoImg->data[i][0] =
		internalVideoImg->data[i][1] =
		internalVideoImg->data[i][2] =
			image->dI[i][0]*0.8 > 255.0f ? 255.0 : image->dI[i][0]*0.8;

	videoImgChanged=true;
}

bool PangolinDSOViewer::needPushDepthImage()
{
    return setting_render_displayDepth;
}
void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{

    if(!setting_render_displayDepth) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
	if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
	last_map = time_now;

	memcpy(internalKFImg->data, image->data, w*h*3);
	kfImgChanged=true;
}
void PangolinDSOViewer::saveAll(){

	printf("saveAll Pressed:\n");
	//Step 1. Save all Keyframes
	std::ofstream poseFile,pathFile;
	char timeStruct[128];
	char poseFileName[256];
	char pathFileName[256];
	char depthDir[256];		
	time_t now_time;
	tm * local;
	now_time=time(NULL);
	local = localtime(&now_time);
	strftime(timeStruct, 128,"%Y-%m-%d-%H-%M", local);
	sprintf(poseFileName,"%s/Pose-%s.txt",dataDir.c_str(), timeStruct);
	sprintf(pathFileName,"%s/Path-%s.txt",dataDir.c_str(), timeStruct);
	sprintf(depthDir,"%s/%s-Depth",dataDir.c_str(), timeStruct);

	char sysCall[256];
	sprintf(sysCall,"mkdir %s",depthDir);
	system(sysCall);


	printf("Pose of keyframes saved to%s\n",poseFileName);	
	poseFile.open(poseFileName);
	for(KeyFrameDisplay* fh : keyframes)
	{
		// save all constraints?
		int fId = fh->id;
		int nId = fh->incoming_id;
		Sophus::Matrix4f m = fh->camToWorld.matrix().cast<float>();						
		float camToWorldMat[16];
		memcpy(camToWorldMat,m.data(),16*sizeof(float));
		poseFile<<fh->id<<" ";
		for(int j=0;j<16;j++){
			poseFile<<camToWorldMat[j]<<" ";
		}
		poseFile<<std::endl;
		char depthFileName[256];
		sprintf(depthFileName,"%s/depth_%06d.txt",depthDir,fId);
		fh->savePC(depthFileName);		
	}
	poseFile.close();	
	printf("keyframe Poses saved\n");

	// save all path

	printf("Path of allframes saved to%s\n",pathFileName);	
	pathFile.open(pathFileName);
	for(unsigned int i=0;i<allFramePoses.size();i++)
	{
		pathFile<<i<<" ";
		for(int j=0;j<3;j++){
			pathFile<<allFramePoses[i][j]<<" ";
		}
		pathFile<<std::endl;
	}
	pathFile.close();	
	printf("Path saved\n");
	//Step 2. Save all Constraints

	

	std::ofstream edgeFile;
	char edgeFileName[256];	
	sprintf(edgeFileName,"%s/%s-EdgeFile.txt",dataDir.c_str(), timeStruct);
	printf("Edges of Graph saved to:%s\n",edgeFileName);
	edgeFile.open(edgeFileName);
	for(unsigned int i=0;i<connections.size();i++)
	{
		if(connections[i].to == 0 || connections[i].from==0) continue;
		int nAct = connections[i].bwdAct + connections[i].fwdAct;
		int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
		if(nAct==0 && nMarg>0)
		{
			edgeFile<<connections[i].from->id<<" "
			        <<connections[i].to->id<<" "
			        <<connections[i].fwdAct<<" "
			        <<connections[i].bwdAct<<" "
					<<connections[i].fwdMarg<<" "
					<<connections[i].bwdMarg<<" "
					<<std::endl;
		}		
	}
	edgeFile.close();


}

void PangolinDSOViewer::readAll(){

	printf("readAll Pressed:\n");

	//Step 1. Save all Keyframes
	std::ifstream poseFile,pathFile;
	char poseFileName[256];
	char pathFileName[256];
	char depthDir[256];
	char seq[30] = "10-25-1";

	sprintf(poseFileName,"/home/wyw/lyh/data/test_seq/%s/pose.txt",seq);
	sprintf(pathFileName,"/home/wyw/lyh/data/test_seq/%s/path.txt",seq);
	sprintf(depthDir,"/home/wyw/lyh/data/test_seq/%s/depth",seq);

	printf("Read from pose of keyframes %s\n",poseFileName);	
	poseFile.open(poseFileName);

	//read the number of items inside depth dir

	std::vector<std::string> allDepthFileNames;

	struct dirent *direntp;
   	DIR *dirp = opendir(depthDir);
 
   	if (dirp != NULL) {
       	while ((direntp = readdir(dirp)) != NULL){
           //printf("%s\n", direntp->d_name);
           allDepthFileNames.push_back(direntp->d_name);
       	}
   	}
   	closedir(dirp);

    int nFrames = allDepthFileNames.size();


    for(int i = 0; i<nFrames; i++){


		KeyFrameDisplay* curKF = new KeyFrameDisplay;
		keyframes.push_back(curKF);
		

		Sophus::Matrix4f m = curKF->camToWorld.matrix().cast<float>();						
		float camToWorldMat[16];
		

		poseFile>>curKF->id;
		for(int j=0;j<16;j++){
			poseFile>>camToWorldMat[j];
		}

		memcpy(m.data(),camToWorldMat,16*sizeof(float));

		curKF->readPC(allDepthFileNames[i].c_str());
		//printf("number of keyframes: %d\n", keyframes.size());

    }
    poseFile.close();

/* Don't need path and edges for now
    //read all path
    allFramePoses.resize(nFrames);

	printf("Read from Path of allframes",pathFileName);	
	pathFile.open(pathFileName);
	for(unsigned int i=0;i<allFramePoses.size();i++)
	{
		int tmpi;
		pathFile>>tmpi;
		for(int j=0;j<3;j++){
			pathFile>>allFramePoses[i][j];
		}
	}
	pathFile.close();	
	printf("Path read finished\n");


	//read all edge file
	std::ifstream edgeFile;
	char edgeFileName[256];	
	sprintf(edgeFileName,"/home/wyw/lyh/data/test_seq/%s-EdgeFile.txt",seq);
	printf("Read Edges of Graph from:%s\n",edgeFileName);
	edgeFile.open(edgeFileName);


	std::vector<std::string> alllines;
	std::string sline;
    while( getline(edgeFile,sline) )
    {    

    	alllines.push_back(sline);

    }



    connections.resize(alllines.size());



	for(unsigned int i=0;i<connections.size();i++){

		std::vector<std::string> strs;
		boost::split(strs,alllines[i],boost::is_any_of(" "));
		
		connections[i].from->id = atoi(strs[0].c_str());	
		connections[i].to->id =  atoi(strs[1].c_str());	
		connections[i].fwdAct =  atoi(strs[2].c_str());	
		connections[i].bwdAct =  atoi(strs[3].c_str());	
		connections[i].fwdMarg =  atoi(strs[4].c_str());	
		connections[i].bwdMarg =  atoi(strs[5].c_str());
					
		
				
	}
	edgeFile.close();
*/

}


void PangolinDSOViewer::publishFrameFromFile(FrameShell* frame, CalibHessian* HCalib, const char* depthFile)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	KeyFrameDisplay* curKF =  new KeyFrameDisplay();
	keyframes.push_back(curKF);
	keyframesByKFID[frame->id] = curKF;
	curKF->setFromF(frame, HCalib);
	curKF->readPC(depthFile);
}

void PangolinDSOViewer::publishObject(FrameShell* frame, CalibHessian* HCalib, const char* modelFile){
	if(!setting_render_display3D) return;
	if(disableAllDisplay) return;
	boost::unique_lock<boost::mutex> lk(model3DMutex);

	KeyFrameDisplay* curKF =  new KeyFrameDisplay();
	objects.push_back(curKF);
	objectsById[frame->id] = curKF;
	curKF->setFromF(frame, HCalib);
	curKF->readObj(modelFile);
}

}
}
